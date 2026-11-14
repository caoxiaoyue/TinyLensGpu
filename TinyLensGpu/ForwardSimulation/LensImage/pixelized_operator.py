"""Operator-based (matrix-free) joint pixelized-source forward simulator.

Provides :class:`PixelizedLensOperator`, a drop-in replacement for
:class:`PixelizedLensSimulator` that avoids building the dense (Nd x Ns)
design matrix.  Uses precomputed lens-operator data and JIT-compiled
primitives for efficient matrix-vector products inside PCG.

"""

from __future__ import annotations

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import jax.scipy.signal as jsp_signal
from jax import Array

from TinyLensGpu.ForwardSimulation.LensImage._pixelized_psf import (
    PSFData,
    PixelizedPSFResponse,
    psf_adjoint,
    psf_forward,
    render_source,
    render_source_adjoint,
)
from TinyLensGpu.ForwardSimulation.LensImage.config import SimulatorConfig
from TinyLensGpu.ForwardSimulation.LensImage.parametric import bin_image_general
from TinyLensGpu.ForwardSimulation.LensImage.pixelized import (
    validate_unit_lens_light_bases,
)
from TinyLensGpu.PhysicalModel.LensImage.composite import PhysicalModel
from TinyLensGpu.PhysicalModel.LensImage.Pixelized.Light import (
    is_pixelized_source_model,
)
from TinyLensGpu.utils.inversion.regularization import (
    FiniteDifferenceRegularization,
    RegData,
    map_source_blocks,
)
from TinyLensGpu.utils.curvature_operator import CurvatureOperator
from TinyLensGpu.utils.lensing.mapping import (
    build_source_grid,
    infer_source_bbox,
    lens_mapping_operator_from,
)

# ==================================================================
# Operator data — precomputed once per bbox, reused across matvecs
# ==================================================================


class LensOperatorData(NamedTuple):
    """Precomputed data for lens mapping operator matvec / rmatvec.

    Stencils are ordered by render subpixel, including the PSF footprint.
    """
    weights: Array       # (N_render_sub, stencil_width) interpolation weights
    indices: Array       # (N_render_sub, stencil_width) flat source-pixel indices
    n_source: int
    nsub: int
    render_parents: Array | None
    n_render: int


class SourceCurvatureData(NamedTuple):
    """Dynamic numerical data for a source-only curvature operator."""

    weights: Array
    indices: Array
    render_parents: Array | None
    psf: PSFData
    noise_variance: Array
    regularization: RegData
    lambda_reg: Array


class SourceCurvatureSpec(NamedTuple):
    """Hashable source topology used as static JIT metadata."""

    psf_shape: tuple[int, int]
    n_source: int
    nsub: int
    n_render: int
    grid_size: int
    regularization_type: str


class JointCurvatureData(NamedTuple):
    """Dynamic numerical data for joint source/lens-light curvature."""

    source: SourceCurvatureData
    lens_matrix: Array
    lens_light_regularization: Array


class JointCurvatureSpec(NamedTuple):
    """Hashable topology for a joint source/lens-light operator."""

    source: SourceCurvatureSpec


@partial(jax.jit, static_argnames=("spec",))
def _source_curvature_kernel(
    source: Array,
    data: SourceCurvatureData,
    spec: SourceCurvatureSpec,
) -> Array:
    """Apply source curvature from typed dynamic data and static topology."""
    fitted = _source_data_forward(source, data, spec)
    return (
        _source_data_adjoint(fitted / data.noise_variance, data, spec)
        + _source_prior_matvec(source, data, spec)
    )


@partial(jax.jit, static_argnames=("spec",))
def _joint_curvature_kernel(
    x: Array,
    data: JointCurvatureData,
    spec: JointCurvatureSpec,
) -> Array:
    """Apply the joint source/lens-light curvature without forming F."""
    source_spec = spec.source
    n_source = source_spec.n_source
    source = x[:n_source]
    lens = x[n_source:]
    fitted = _source_data_forward(source, data.source, source_spec) + data.lens_matrix @ lens
    weighted = fitted / data.source.noise_variance
    source_out = (
        _source_data_adjoint(weighted, data.source, source_spec)
        + _source_prior_matvec(source, data.source, source_spec)
    )
    lens_out = data.lens_matrix.T @ weighted + data.lens_light_regularization * lens
    return jnp.concatenate([source_out, lens_out])


def _source_data_forward(
    source: Array,
    data: SourceCurvatureData,
    spec: SourceCurvatureSpec,
) -> Array:
    brightness = render_source(
        source, data.weights, data.indices, data.render_parents, spec.nsub, spec.n_render,
    )
    return psf_forward(brightness, data.psf, spec.psf_shape)


def _source_data_adjoint(
    weighted_active: Array,
    data: SourceCurvatureData,
    spec: SourceCurvatureSpec,
) -> Array:
    return render_source_adjoint(
        psf_adjoint(weighted_active, data.psf, spec.psf_shape),
        data.weights, data.indices, data.render_parents, spec.nsub, spec.n_source,
    )


# ==================================================================
# JIT-compiled operator primitives  (no ``self`` — pure array args)
# ==================================================================


def _geom_mean(a: Array, b: Array) -> Array:
    """Geometric mean, clipped away from zero."""
    return jnp.sqrt(jnp.maximum(a, 1e-30) * jnp.maximum(b, 1e-30))


def _geom_mean3(a: Array, b: Array, c: Array) -> Array:
    """Geometric mean of three arrays."""
    return jnp.exp(
        (jnp.log(jnp.maximum(a, 1e-30))
         + jnp.log(jnp.maximum(b, 1e-30))
         + jnp.log(jnp.maximum(c, 1e-30))) / 3.0
    )


def _weighted_first_order_matvec_jit(
    s: Array, scale: Array | None, scale_factor: Array, n: int
) -> Array:
    """Edge-weighted first-order Laplacian ``R @ s``."""
    s_2d = s.reshape(n, n)
    scale_2d = scale.reshape(n, n) if scale is not None else None

    out_x = jnp.zeros_like(s_2d)
    out_y = jnp.zeros_like(s_2d)
    if n > 1:
        if scale_2d is None:
            w_x = jnp.ones((n, n - 1), dtype=s.dtype)
        else:
            w_x = _geom_mean(scale_2d[:, :-1], scale_2d[:, 1:])
        diff_x = s_2d[:, 1:] - s_2d[:, :-1]
        wdiff_x = w_x * diff_x
        out_x = out_x.at[:, :-1].add(-wdiff_x)
        out_x = out_x.at[:, 1:].add(wdiff_x)
        out_x = out_x.at[:, -1].add(s_2d[:, -1])
    if n > 1:
        if scale_2d is None:
            w_y = jnp.ones((n - 1, n), dtype=s.dtype)
        else:
            w_y = _geom_mean(scale_2d[:-1, :], scale_2d[1:, :])
        diff_y = s_2d[1:, :] - s_2d[:-1, :]
        wdiff_y = w_y * diff_y
        out_y = out_y.at[:-1, :].add(-wdiff_y)
        out_y = out_y.at[1:, :].add(wdiff_y)
        out_y = out_y.at[-1, :].add(s_2d[-1, :])
    return (scale_factor * (out_x + out_y)).ravel()


def _weighted_second_order_matvec_jit(
    s: Array, scale: Array | None, scale_factor: Array, n: int
) -> Array:
    """Edge-weighted second-order curvature ``R @ s``."""
    s_2d = s.reshape(n, n)
    scale_2d = scale.reshape(n, n) if scale is not None else None

    out_x = jnp.zeros_like(s_2d)
    out_y = jnp.zeros_like(s_2d)
    if n > 1:
        if scale_2d is None:
            w_near_x = jnp.ones((n,), dtype=s.dtype)
        else:
            w_near_x = _geom_mean(scale_2d[:, -2], scale_2d[:, -1])
        diff_near_x = s_2d[:, -1] - s_2d[:, -2]
        out_x = out_x.at[:, -2].add(-w_near_x * diff_near_x)
        out_x = out_x.at[:, -1].add(w_near_x * diff_near_x)
    if n > 2:
        if scale_2d is None:
            w_x2 = jnp.ones((n, n - 2), dtype=s.dtype)
        else:
            w_x2 = _geom_mean3(scale_2d[:, :-2], scale_2d[:, 1:-1], scale_2d[:, 2:])
        c_x = s_2d[:, :-2] - 2.0 * s_2d[:, 1:-1] + s_2d[:, 2:]
        wc_x = w_x2 * c_x
        out_x = out_x.at[:, :-2].add(wc_x)
        out_x = out_x.at[:, 1:-1].add(-2.0 * wc_x)
        out_x = out_x.at[:, 2:].add(wc_x)
    if n > 1:
        out_x = out_x.at[:, -1].add(s_2d[:, -1])

    if n > 1:
        if scale_2d is None:
            w_near_y = jnp.ones((n,), dtype=s.dtype)
        else:
            w_near_y = _geom_mean(scale_2d[-2, :], scale_2d[-1, :])
        diff_near_y = s_2d[-1, :] - s_2d[-2, :]
        out_y = out_y.at[-2, :].add(-w_near_y * diff_near_y)
        out_y = out_y.at[-1, :].add(w_near_y * diff_near_y)
    if n > 2:
        if scale_2d is None:
            w_y2 = jnp.ones((n - 2, n), dtype=s.dtype)
        else:
            w_y2 = _geom_mean3(scale_2d[:-2, :], scale_2d[1:-1, :], scale_2d[2:, :])
        c_y = s_2d[:-2, :] - 2.0 * s_2d[1:-1, :] + s_2d[2:, :]
        wc_y = w_y2 * c_y
        out_y = out_y.at[:-2, :].add(wc_y)
        out_y = out_y.at[1:-1, :].add(-2.0 * wc_y)
        out_y = out_y.at[2:, :].add(wc_y)
    if n > 1:
        out_y = out_y.at[-1, :].add(s_2d[-1, :])

    return (scale_factor * (out_x + out_y)).ravel()


def _source_prior_matvec(s: Array, data: SourceCurvatureData, spec: SourceCurvatureSpec) -> Array:
    """Apply the source prior in the same normal operator as the PSF response."""
    scale, scale_factor = data.regularization
    n, reg_type = spec.grid_size, spec.regularization_type
    if reg_type == "zero-order":
        reg_term = scale * s if scale is not None else s
    elif reg_type == "first-order":
        reg_term = _weighted_first_order_matvec_jit(s, scale, scale_factor, n)
    elif reg_type == "second-order":
        reg_term = _weighted_second_order_matvec_jit(s, scale, scale_factor, n)
    else:
        # Unreachable: reg_type is a static arg validated in __init__ to be
        # one of {zero, first, second}-order.  Kept for type-narrowing.
        raise ValueError(f"Unsupported reg_type for operator backend: {reg_type!r}")
    return data.lambda_reg * reg_term


# ==================================================================
# Main class
# ==================================================================


class PixelizedLensOperator:
    """Matrix-free simulator for a pixelized source and optional lens light.

    Parameters
    ----------
    phys_model : PhysicalModel
        Model with lens mass and exactly one pixelized source.
    sim_config : SimulatorConfig
        Image grid, PSF, mask, and subsampling configuration.
    detach_bbox : bool, optional
        When True (default), stop_gradient is applied to square bbox bounds.
    """

    def __init__(
        self,
        phys_model: PhysicalModel,
        sim_config: SimulatorConfig,
        detach_bbox: bool = True,
    ) -> None:
        self.phys_model = phys_model
        self.sim_config = sim_config
        self.nsub = int(sim_config.nsub)
        self.detach_bbox = detach_bbox

        n_pixelized = sum(
            is_pixelized_source_model(s) for s in phys_model.source_light
        )
        if n_pixelized != len(phys_model.source_light):
            raise ValueError("PixelizedLensOperator does not support a parametric source")
        if n_pixelized != 1:
            raise ValueError("PixelizedLensOperator requires a single pixelized source")

        source = phys_model.source_light[0]
        self.source_model = source
        _reg_type = str(source.regularization_type)
        if _reg_type not in ("zero-order", "first-order", "second-order"):
            raise ValueError(
                f"PixelizedLensOperator only supports finite-difference "
                f"regularization types (zero-order, first-order, second-order), "
                f"got {_reg_type!r}."
            )
        self.reg_type = _reg_type
        self.source_n = int(source.n)
        self.n_source_pixels = self.source_n * self.source_n

        self.n_lens_light = len(phys_model.lens_light)
        self.has_lens_light = self.n_lens_light > 0
        validate_unit_lens_light_bases(phys_model)

        self.image_shape = tuple(sim_config.mask.shape)
        H, W = self.image_shape
        self.n_full_pixels = H * W

        self.mask = jnp.asarray(sim_config.mask, dtype=bool)
        self.active_mask = ~self.mask
        self.flat_indices = jnp.flatnonzero(self.active_mask.ravel())
        self.n_active = int(self.flat_indices.shape[0])

        self.image_x_active = jnp.asarray(sim_config.xgrid)[self.active_mask]
        self.image_y_active = jnp.asarray(sim_config.ygrid)[self.active_mask]
        self.psf_kernel = jnp.asarray(sim_config.psf_kernel)
        self._psf_response = PixelizedPSFResponse(sim_config.mask, self.psf_kernel, self.nsub)
        self._render_coords = self._psf_response.ray_coordinates(sim_config)
        self._render_native_coords = (
            sim_config.xgrid.ravel()[self._psf_response.render_indices],
            sim_config.ygrid.ravel()[self._psf_response.render_indices],
        )

        # Source seed mask
        self.source_seed_mask = jnp.asarray(sim_config.source_seed_mask, dtype=bool)
        self.seed_active_mask = ~self.source_seed_mask
        if not jnp.all(self.seed_active_mask <= self.active_mask):
            raise ValueError(
                "source_seed_mask active region must be a subset of mask active region."
            )
        self.seed_flat_indices = jnp.flatnonzero(self.seed_active_mask.ravel())
        self.image_x_seed = jnp.asarray(sim_config.xgrid)[self.seed_active_mask]
        self.image_y_seed = jnp.asarray(sim_config.ygrid)[self.seed_active_mask]

        self.image_shape_sub = sim_config.subgrid_shape
        self.flat_indices_sub = sim_config.flat_indices
        self.image_x_active_sub = sim_config.xgrid_sub_1d
        self.image_y_active_sub = sim_config.ygrid_sub_1d

        self._source_curvature_spec = SourceCurvatureSpec(
            psf_shape=self._psf_response.fft_shape,
            n_source=self.n_source_pixels,
            nsub=self.nsub,
            n_render=self._psf_response.n_render,
            grid_size=self.source_n,
            regularization_type=self.reg_type,
        )

    @property
    def psf_response(self) -> PixelizedPSFResponse:
        """Shared fitted/rendered image response and source-stencil operations."""
        return self._psf_response

    # ------------------------------------------------------------------
    # Bbox helpers
    # ------------------------------------------------------------------

    def _get_beta_sub_and_seed(self):
        x, y = self._render_coords
        beta_x_sub, beta_y_sub = self.phys_model.deflection(x=x, y=y)
        beta_x_seed, beta_y_seed = self.phys_model.deflection(
            x=self.image_x_seed, y=self.image_y_seed
        )
        return beta_x_sub, beta_y_sub, beta_x_seed, beta_y_seed

    def _infer_and_fix_bbox(self, beta_x_seed, beta_y_seed):
        xmin, xmax, ymin, ymax = infer_source_bbox(
            beta_x_seed, beta_y_seed,
            padding=self.sim_config.source_bbox_padding,
            outlier_frac=self.sim_config.source_bbox_outlier_frac,
        )
        if self.detach_bbox:
            xmin = jax.lax.stop_gradient(xmin)
            xmax = jax.lax.stop_gradient(xmax)
            ymin = jax.lax.stop_gradient(ymin)
            ymax = jax.lax.stop_gradient(ymax)
        return xmin, xmax, ymin, ymax

    def ray_trace_seed(self) -> tuple[Array, Array]:
        """Ray-trace seed mask pixels to the source plane.

        Returns ``(beta_x_seed, beta_y_seed)`` for all active pixels in the
        source seed mask.
        """
        beta_x_seed, beta_y_seed = self.phys_model.deflection(
            x=self.image_x_seed, y=self.image_y_seed
        )
        return beta_x_seed, beta_y_seed

    # ------------------------------------------------------------------
    # Precompute lens-operator data (called ONCE per bbox)
    # ------------------------------------------------------------------

    def precompute_operator_data(
        self, xmin, xmax, ymin, ymax, *,
        _betas_sub: tuple | None = None,
    ) -> LensOperatorData:
        """Ray-trace and compute weights and indices for source interpolation.

        Prepare once per geometry and reuse for all matvecs.

        If ``_betas_sub`` is provided, it must cover the render subpixels in
        their fixed order; the deflection call is then skipped.
        """
        if _betas_sub is not None:
            beta_x_sub, beta_y_sub = _betas_sub
            if beta_x_sub.shape != self._render_coords[0].shape or beta_y_sub.shape != beta_x_sub.shape:
                raise ValueError("Precomputed rays must cover the render subpixels")
        else:
            x, y = self._render_coords
            beta_x_sub, beta_y_sub = self.phys_model.deflection(x=x, y=y)
        source_x_axis, source_y_axis, _, _ = build_source_grid(
            self.source_n, xmin, xmax, ymin, ymax,
        )
        data_mesh = jnp.stack(
            [jnp.ravel(beta_x_sub), jnp.ravel(beta_y_sub)], axis=1
        )
        weights, indices, _ = lens_mapping_operator_from(
            data_mesh,
            source_x_axis[0],
            source_x_axis[-1],
            source_y_axis[0],
            source_y_axis[-1],
            self.source_n,
            interpolation=getattr(self.source_model, "interpolation", "bilinear"),
        )
        return LensOperatorData(
            weights=weights,
            indices=indices,
            n_source=self.n_source_pixels,
            nsub=self.nsub,
            render_parents=self._psf_response.render_parents,
            n_render=self._psf_response.n_render,
        )

    # ------------------------------------------------------------------
    # Forward model (with optional precomputed data)
    # ------------------------------------------------------------------

    def forward_model(
        self, s: Array, xmin, xmax, ymin, ymax,
        op_data: LensOperatorData | None = None,
    ) -> Array:
        """Render, convolve and select the native fitted pixels."""
        if op_data is None:
            op_data = self.precompute_operator_data(xmin, xmax, ymin, ymax)
        brightness = render_source(
            jnp.asarray(s), op_data.weights, op_data.indices,
            op_data.render_parents, op_data.nsub, op_data.n_render,
        )
        return self._psf_response.forward(brightness)

    def build_lens_light_matrix(self) -> Array:
        """Return the PSF-convolved unit-amplitude lens-light basis matrix."""
        if not self.has_lens_light:
            return jnp.zeros((self.n_active, 0), dtype=self.psf_kernel.dtype)
        if self.nsub > 1:
            xgrid = jnp.asarray(self.sim_config.xgrid_sub)
            ygrid = jnp.asarray(self.sim_config.ygrid_sub)
        else:
            xgrid = jnp.asarray(self.sim_config.xgrid)
            ygrid = jnp.asarray(self.sim_config.ygrid)
        images = jnp.stack([
            component.light(x=xgrid, y=ygrid)
            for component in self.phys_model.lens_light
        ], axis=-1)
        if self.nsub > 1:
            images = bin_image_general(images, self.nsub)
        convolved = jax.vmap(
            lambda image: jsp_signal.fftconvolve(
                image, self.psf_kernel, mode="same"
            ),
            in_axes=-1,
            out_axes=-1,
        )(images)
        return convolved.reshape(self.n_full_pixels, self.n_lens_light)[
            self.flat_indices
        ]

    def build_joint_system(
        self,
        noise_1d: Array,
        data_1d: Array,
        xmin,
        xmax,
        ymin,
        ymax,
        lambda_reg: Array,
        reg_data: RegData,
        lens_light_regularization: Array,
        op_data: LensOperatorData | None = None,
        image_adjoint: Array | None = None,
        source_rhs: Array | None = None,
        lens_matrix: Array | None = None,
    ) -> tuple[CurvatureOperator, Array, Array]:
        """Build the joint curvature operator, right-hand side, and lens matrix."""
        source_operator = self.build_source_curvature_operator(
            noise_1d,
            xmin,
            xmax,
            ymin,
            ymax,
            lambda_reg,
            reg_data,
            op_data=op_data,
        )
        if lens_matrix is None:
            lens_matrix = self.build_lens_light_matrix()
        eps = jnp.asarray(lens_light_regularization, dtype=lens_matrix.dtype)
        joint_data = JointCurvatureData(
            source=source_operator.data,
            lens_matrix=lens_matrix,
            lens_light_regularization=eps,
        )
        joint_operator = CurvatureOperator(
            data=joint_data,
            kernel=_joint_curvature_kernel,
            spec=JointCurvatureSpec(source=self._source_curvature_spec),
            size=self.n_source_pixels + self.n_lens_light,
        )
        if source_rhs is None:
            source_rhs = self.build_rhs(
                data_1d,
                noise_1d,
                xmin,
                xmax,
                ymin,
                ymax,
                op_data=op_data,
                image_adjoint=image_adjoint,
            )
        lens_rhs = lens_matrix.T @ (jnp.asarray(data_1d) / (jnp.asarray(noise_1d) ** 2))
        return joint_operator, jnp.concatenate([source_rhs, lens_rhs]), lens_matrix

    def reconstruct_components(
        self, source_pixels: Array, lens_light_intensities: Array,
        xmin, xmax, ymin, ymax,
        op_data: LensOperatorData | None = None,
        lens_matrix: Array | None = None,
    ) -> tuple[Array, Array]:
        """Return active-pixel lensed-source and lens-light images."""
        source_image = self.forward_model(
            source_pixels, xmin, xmax, ymin, ymax, op_data=op_data,
        )
        if lens_matrix is None:
            lens_matrix = self.build_lens_light_matrix()
        lens_image = lens_matrix @ jnp.asarray(lens_light_intensities)
        return source_image, lens_image

    # ------------------------------------------------------------------
    # Right-hand side: b = Mᵀ C⁻¹ d
    # ------------------------------------------------------------------

    def prepare_observation(self, data_1d: Array, noise_1d: Array):
        """Precompute the data adjoint and the preconditioner's noise weights."""
        image_adjoint = self._psf_response.adjoint(data_1d / noise_1d**2)
        return image_adjoint, self._effective_noise_weights(noise_1d)

    def _effective_noise_weights(self, noise_1d):
        return self._psf_response.precision_diagonal(1.0 / noise_1d**2)

    def build_rhs(
        self,
        data_1d: Array,
        noise_1d: Array,
        xmin,
        xmax,
        ymin,
        ymax,
        op_data: LensOperatorData | None = None,
        image_adjoint: Array | None = None,
    ) -> Array:
        """Compute b = Lᵀ Bᵀ C⁻¹ d, optionally reusing the fixed data adjoint."""
        if op_data is None:
            op_data = self.precompute_operator_data(xmin, xmax, ymin, ymax)
        if image_adjoint is None:
            weighted = jnp.asarray(data_1d) / jnp.asarray(noise_1d)**2
            image_adjoint = self._psf_response.adjoint(weighted)
        return render_source_adjoint(
            image_adjoint, op_data.weights, op_data.indices,
            op_data.render_parents, op_data.nsub, op_data.n_source,
        )

    # ------------------------------------------------------------------
    # A-operator: A(s) = Mᵀ C⁻¹ M s + λ R s   (Ns → Ns)
    # ------------------------------------------------------------------

    def build_source_curvature_operator(
        self,
        noise_1d: Array,
        xmin, xmax, ymin, ymax,
        lambda_reg: Array,
        reg_data: RegData,
        op_data: LensOperatorData | None = None,
    ) -> CurvatureOperator:
        """Build source curvature as a typed JAX PyTree operator.

        ``reg_data`` is a :class:`~TinyLensGpu.utils.inversion.regularization.RegData`
        tuple holding the per-pixel adaptive ``scale`` array and physical spacing
        factors for the edge-weighted finite-difference regularisation term.
        """
        if op_data is None:
            op_data = self.precompute_operator_data(xmin, xmax, ymin, ymax)

        data = SourceCurvatureData(
            weights=op_data.weights,
            indices=op_data.indices,
            render_parents=op_data.render_parents,
            psf=self._psf_response.data,
            noise_variance=jnp.asarray(noise_1d) ** 2,
            regularization=reg_data,
            lambda_reg=jnp.asarray(lambda_reg),
        )
        return CurvatureOperator(
            data=data,
            kernel=_source_curvature_kernel,
            spec=self._source_curvature_spec,
            size=self.n_source_pixels,
        )

    def apply_source_data_adjoint(
        self,
        weighted_active: Array,
        data: SourceCurvatureData,
    ) -> Array:
        """Apply the source data adjoint without exposing mapping internals."""
        return _source_data_adjoint(
            jnp.asarray(weighted_active), data, self._source_curvature_spec,
        )

    # ------------------------------------------------------------------
    # Block-diagonal preconditioner
    # ------------------------------------------------------------------

    def build_data_blocks(
        self,
        noise_1d,
        xmin,
        xmax,
        ymin,
        ymax,
        block_size=10,
        noise_weights=None,
    ):
        """Build native-resolution data blocks and their source pixel indices.

        This approximation uses the selected interpolation at native
        resolution even when the curvature operator integrates a sub-grid.
        """
        if noise_weights is None:
            noise_weights = self._effective_noise_weights(jnp.asarray(noise_1d))
        x, y = self._render_native_coords
        beta_x, beta_y = self.phys_model.deflection(x=x, y=y)
        weights, indices, _ = lens_mapping_operator_from(
            jnp.stack([beta_x.ravel(), beta_y.ravel()], axis=1),
            xmin,
            xmax,
            ymin,
            ymax,
            self.source_n,
            interpolation=getattr(self.source_model, "interpolation", "bilinear"),
        )

        def build(x, xe, y, ye, pixels):
            local = -jnp.ones(self.n_source_pixels, dtype=jnp.int32)
            local = local.at[pixels].set(jnp.arange(pixels.size, dtype=jnp.int32))
            neighbors = local[indices]
            valid = (neighbors[:, :, None] >= 0) & (neighbors[:, None, :] >= 0)
            rows = jnp.broadcast_to(neighbors[:, :, None], valid.shape)
            cols = jnp.broadcast_to(neighbors[:, None, :], valid.shape)
            values = (
                weights[:, :, None] * weights[:, None, :] * noise_weights[:, None, None]
            )
            block = jnp.zeros((pixels.size, pixels.size), dtype=weights.dtype)
            # Drop cross-block pairs to avoid concentrated zero updates at (0, 0).
            return block.at[
                jnp.where(valid, rows, pixels.size).ravel(),
                jnp.where(valid, cols, 0).ravel(),
            ].add(values.astype(block.dtype).ravel(), mode="drop")

        return map_source_blocks(self.source_n, block_size, build)

    def build_block_diag_preconditioner(
        self,
        noise_1d: Array,
        xmin,
        xmax,
        ymin,
        ymax,
        lambda_reg: Array,
        reg_builder: FiniteDifferenceRegularization,
        block_size: int = 10,
        scale: Array | None = None,
        *,
        noise_weights=None,
        data_blocks=None,
        reg_blocks=None,
    ) -> tuple[Array | list[Array], Array | list[Array]]:
        """Factor G_k + lambda R_k, reusing fixed blocks when available.

        Uniform grids are stacked; ragged edge blocks retain their individual
        shapes. Jitter and the native-mapping approximation are unchanged.
        """
        if data_blocks is None:
            data_blocks = self.build_data_blocks(
                noise_1d,
                xmin,
                xmax,
                ymin,
                ymax,
                block_size,
                noise_weights,
            )
        if reg_blocks is None:
            reg_blocks = reg_builder.block_matrices(
                xmin,
                xmax,
                ymin,
                ymax,
                scale=scale,
                block_size=block_size,
            )
        matrices, masks = data_blocks

        def factor(data, regularization):
            block = data + lambda_reg * regularization
            block = 0.5 * (block + block.T)
            jitter = 1e-6 * jnp.maximum(jnp.mean(jnp.abs(jnp.diag(block))), 1e-8)
            return jnp.linalg.cholesky(
                block + jitter * jnp.eye(block.shape[0], dtype=block.dtype)
            )

        if isinstance(matrices, (list, tuple)):
            chols = [factor(data, reg) for data, reg in zip(matrices, reg_blocks)]
        else:
            chols = jax.vmap(factor)(matrices, reg_blocks)
        return chols, masks

    # ------------------------------------------------------------------
    # Legacy: dense preconditioner  (for testing / small grids)
    # ------------------------------------------------------------------
    @staticmethod
    def logdet_block_diag(block_chols: Array | list[Array]) -> Array:
        """Compute ``log|P|`` from block-diagonal Cholesky factors."""
        if not isinstance(block_chols, (list, tuple)):
            return 2.0 * jnp.sum(jnp.log(jnp.diagonal(block_chols, axis1=-2, axis2=-1)))

        if len(block_chols) == 0:
            return jnp.array(0.0)

        total = jnp.array(0.0, dtype=block_chols[0].dtype)
        for chol in block_chols:
            total = total + 2.0 * jnp.sum(jnp.log(jnp.diag(chol)))
        return total

    # ------------------------------------------------------------------
    # design_matrix — for parity testing only
    # ------------------------------------------------------------------

    def design_matrix(self) -> tuple[Array, tuple]:
        """Build F (Nd × Ns) by applying forward_model to each basis vector.

        **Slow** — for verification only.
        """
        _, _, beta_x_seed, beta_y_seed = self._get_beta_sub_and_seed()
        xmin, xmax, ymin, ymax = self._infer_and_fix_bbox(
            beta_x_seed, beta_y_seed
        )
        eye = jnp.eye(self.n_source_pixels, dtype=self.psf_kernel.dtype)
        columns = [self.forward_model(eye[i], xmin, xmax, ymin, ymax)
                   for i in range(self.n_source_pixels)]
        F = jnp.stack(columns, axis=1)
        return F, (xmin, xmax, ymin, ymax)

    def __repr__(self) -> str:
        return (
            f"PixelizedLensOperator(image_shape={self.image_shape}, "
            f"source_shape=({self.source_n}, {self.source_n}), "
            f"n_active={self.n_active})"
        )


__all__ = ["PixelizedLensOperator"]
