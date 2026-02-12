"""
Forward model for pixelized source gravitational lensing.

This module provides the forward modeling components for pixelized source
reconstruction, including ray-tracing, mesh generation, and mapping matrix
construction.
"""

from __future__ import annotations

import functools
import jax.numpy as jnp
from jax import jit, Array
import numpy as np
from typing import Optional, Tuple, Union

from .config import make_grid_2d
from TinyLensGpu.PhysicalModel.LensImage.composite import PhysicalModel
from TinyLensGpu.PhysicalModel.LensImage.Pixelized import PixelizedSourceModel
from TinyLensGpu.utils.lensing import (
    lens_mapping_matrix_from,
    lens_mapping_matrix_bilinear_rectangular_from,
    lens_mapping_operator_bilinear_rectangular_from,
    regularization_sparse_rectangular_from,
    sparse_regularization_dense_from,
    apply_psf_to_mapping_matrix,
)
from TinyLensGpu.utils.interpolation.kernels import get_interpolation_weights
from TinyLensGpu.utils.mesh import sample_points_weighted
from TinyLensGpu.utils.inversion import (
    LinearInversion,
    OperatorInversion,
    OperatorNNLSInversion,
    NNLSInversion,
)


def _normalize_inversion_backend(name: str) -> str:
    """Normalize user-facing backend aliases into canonical backend names.

    Parameters
    ----------
    name : str
        Backend identifier provided by API callers. Supported names are
        ``"matrix"`` and ``"operator"`` plus legacy aliases
        ``"exact"`` and ``"fast"``.

    Returns
    -------
    str
        Canonical backend name (``"matrix"`` or ``"operator"``).
    """
    backend = str(name).strip().lower()
    if backend in {"exact", "matrix"}:
        return "matrix"
    if backend in {"fast", "operator"}:
        return "operator"
    raise ValueError(
        f"Unknown inversion_backend='{name}'. Expected one of: "
        "'matrix', 'operator' (legacy aliases: 'exact', 'fast')."
    )


class PixelizedLensSimulator:
    def __init__(
        self,
        image_data: np.ndarray,
        dpix: float,
        phys_model: PhysicalModel,
        psf_kernel: np.ndarray,
        mask: Optional[np.ndarray] = None,
        lensed_source_image: Optional[np.ndarray] = None,
    ) -> None:
        
        self.dpix = dpix
        self.npix = image_data.shape[0]
        self.phys_model = phys_model
        # Use the centralized extraction method
        self.pix_src_model = self.phys_model.get_pixelized_source_model()
        self.psf_kernel = jnp.array(psf_kernel)
        
        if mask is None:
            mask = np.zeros_like(image_data, dtype=bool)
        self.mask = jnp.array(mask)
        
        if lensed_source_image is None:
            lensed_source_image = np.ones_like(image_data)
        self.lensed_source_image = jnp.array(lensed_source_image)
        
        xgrid_2d, ygrid_2d = make_grid_2d(self.npix, self.dpix)
        self.xgrid_unmask = jnp.array(xgrid_2d[~mask], dtype=jnp.float32)
        self.ygrid_unmask = jnp.array(ygrid_2d[~mask], dtype=jnp.float32)
        
        # Precompute unmasked indices for PSF application
        # mask is boolean, True where masked. We want indices of False (unmasked).
        y_indices, x_indices = np.where(~mask)
        self.unmasked_indices = (jnp.array(y_indices), jnp.array(x_indices))
        self.image_shape = (self.npix, self.npix)

        psf_h, psf_w = self.psf_kernel.shape
        self.psf_shape = (int(psf_h), int(psf_w))
        self.fft_shape = (self.npix + int(psf_h) - 1, self.npix + int(psf_w) - 1)
        self.psf_fft = jnp.fft.rfft2(self.psf_kernel, s=self.fft_shape)

        self._cached_operator_weights = None
        self._cached_operator_indices = None
        self._cached_operator_signature = None

        self.source_grid_shape = None
        self.source_grid_bounds = None
        
        self._generate_source_mesh(self.lensed_source_image, self.mask)
    
    def _generate_source_mesh(self, lensed_source_image: Array | np.ndarray, mask: Array | np.ndarray) -> None:
        model = self.pix_src_model

        if getattr(model, "is_rectangular_grid", False):
            data_mesh_beta = np.asarray(self.data_mesh_beta, dtype=np.float32)
            explicit_bounds = getattr(model, "source_grid_bounds", None)
            if explicit_bounds is None:
                x_min = float(np.min(data_mesh_beta[:, 0]))
                x_max = float(np.max(data_mesh_beta[:, 0]))
                y_min = float(np.min(data_mesh_beta[:, 1]))
                y_max = float(np.max(data_mesh_beta[:, 1]))
                margin_frac = float(getattr(model, "source_grid_margin_frac", 0.1))

                x_span = max(x_max - x_min, 1e-5)
                y_span = max(y_max - y_min, 1e-5)
                x_margin = margin_frac * x_span
                y_margin = margin_frac * y_span
                x_min -= x_margin
                x_max += x_margin
                y_min -= y_margin
                y_max += y_margin
            else:
                x_min, x_max, y_min, y_max = [float(v) for v in explicit_bounds]

            nx = int(getattr(model, "source_grid_nx", 64))
            ny = int(getattr(model, "source_grid_ny", 64))

            x_lin = jnp.linspace(x_min, x_max, nx, dtype=jnp.float32)
            y_lin = jnp.linspace(y_min, y_max, ny, dtype=jnp.float32)
            xx, yy = jnp.meshgrid(x_lin, y_lin, indexing='xy')
            source_mesh = jnp.stack([xx.reshape(-1), yy.reshape(-1)], axis=1)

            self.source_mesh = source_mesh
            self.source_grid_shape = (ny, nx)
            self.source_grid_bounds = (x_min, x_max, y_min, y_max)
            return
        
        source_mesh, (H, W), _ = sample_points_weighted(
            img=np.array(lensed_source_image),
            mask=~np.array(mask),
            n_points=model.n_source_points,
            alpha=model.mesh_alpha,
            blur_sigma_px=model.mesh_blur_sigma,
            replace=False,
            normalize_xy=False,
            pixel_jitter=False,
            method=model.mesh_method,
            seed=model.mesh_seed,
        )
        
        source_mesh = (source_mesh - np.array([(W-1)/2, (H-1)/2])) * self.dpix
        self.source_mesh = jnp.array(source_mesh, dtype=jnp.float32)
        self.source_grid_shape = None
        self.source_grid_bounds = None
    
    @functools.partial(jit, static_argnums=(0,))
    def ray_trace(self, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        beta_x, beta_y = self.phys_model.deflection(x=x, y=y)
        return jnp.stack([beta_x, beta_y], axis=1)

    @property
    def source_mesh_beta(self) -> jnp.ndarray:
        if getattr(self.pix_src_model, "is_rectangular_grid", False):
            return self.source_mesh
        return self.ray_trace(self.source_mesh[:, 0], self.source_mesh[:, 1])
    
    @property
    def data_mesh_beta(self) -> jnp.ndarray:
        return self.ray_trace(self.xgrid_unmask, self.ygrid_unmask)
    
    def build_lens_mapping_matrix(self) -> jnp.ndarray:
        if getattr(self.pix_src_model, "is_rectangular_grid", False):
            if self.source_grid_shape is None or self.source_grid_bounds is None:
                raise RuntimeError("Rectangular source grid metadata missing.")
            ny, nx = self.source_grid_shape
            x_min, x_max, y_min, y_max = self.source_grid_bounds
            return lens_mapping_matrix_bilinear_rectangular_from(
                data_mesh_beta=self.data_mesh_beta,
                x_min=x_min,
                x_max=x_max,
                y_min=y_min,
                y_max=y_max,
                nx=nx,
                ny=ny,
            )

        return lens_mapping_matrix_from(
            source_mesh_beta=self.source_mesh_beta,
            data_mesh_beta=self.data_mesh_beta,
            k_neighbors=self.pix_src_model.k_neighbors,
            kernel=self.pix_src_model.interp_kernel,
            radius_scale=self.pix_src_model.radius_scale,
        )

    def _operator_signature_from(self, source_mesh_beta: jnp.ndarray, data_mesh_beta: jnp.ndarray) -> Tuple:
        if getattr(self.pix_src_model, "is_rectangular_grid", False):
            return (
                "rectangular_bilinear",
                tuple(data_mesh_beta.shape),
                float(np.asarray(data_mesh_beta, dtype=np.float32).sum()),
                tuple(self.source_grid_shape) if self.source_grid_shape is not None else None,
                tuple(self.source_grid_bounds) if self.source_grid_bounds is not None else None,
            )

        source_np = np.asarray(source_mesh_beta, dtype=np.float32)
        data_np = np.asarray(data_mesh_beta, dtype=np.float32)
        return (
            tuple(source_np.shape),
            tuple(data_np.shape),
            float(source_np.sum()),
            float((source_np * source_np).sum()),
            float(data_np.sum()),
            float((data_np * data_np).sum()),
            int(self.pix_src_model.k_neighbors),
            str(self.pix_src_model.interp_kernel),
            float(self.pix_src_model.radius_scale),
        )

    def clear_operator_cache(self) -> None:
        self._cached_operator_weights = None
        self._cached_operator_indices = None
        self._cached_operator_signature = None

    def build_lens_mapping_operator(
        self,
        *,
        use_cache: bool = True,
        cache_policy: str = "safe",
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        policy = str(cache_policy).strip().lower()

        if getattr(self.pix_src_model, "is_rectangular_grid", False):
            source_mesh_beta = self.source_mesh_beta
            data_mesh_beta = self.data_mesh_beta
            if self.source_grid_shape is None or self.source_grid_bounds is None:
                raise RuntimeError("Rectangular source grid metadata missing.")
            ny, nx = self.source_grid_shape
            x_min, x_max, y_min, y_max = self.source_grid_bounds

            if use_cache and policy == "unsafe_static":
                if self._cached_operator_weights is not None and self._cached_operator_indices is not None:
                    return self._cached_operator_weights, self._cached_operator_indices
                weights, indices, _ = lens_mapping_operator_bilinear_rectangular_from(
                    data_mesh_beta=data_mesh_beta,
                    x_min=x_min,
                    x_max=x_max,
                    y_min=y_min,
                    y_max=y_max,
                    nx=nx,
                    ny=ny,
                )
                self._cached_operator_weights = weights
                self._cached_operator_indices = indices
                self._cached_operator_signature = "unsafe_static"
                return weights, indices

            if use_cache and policy == "safe":
                signature = self._operator_signature_from(source_mesh_beta, data_mesh_beta)
                if (
                    self._cached_operator_weights is not None
                    and self._cached_operator_indices is not None
                    and self._cached_operator_signature == signature
                ):
                    return self._cached_operator_weights, self._cached_operator_indices

                weights, indices, _ = lens_mapping_operator_bilinear_rectangular_from(
                    data_mesh_beta=data_mesh_beta,
                    x_min=x_min,
                    x_max=x_max,
                    y_min=y_min,
                    y_max=y_max,
                    nx=nx,
                    ny=ny,
                )
                self._cached_operator_weights = weights
                self._cached_operator_indices = indices
                self._cached_operator_signature = signature
                return weights, indices

            weights, indices, _ = lens_mapping_operator_bilinear_rectangular_from(
                data_mesh_beta=data_mesh_beta,
                x_min=x_min,
                x_max=x_max,
                y_min=y_min,
                y_max=y_max,
                nx=nx,
                ny=ny,
            )
            return weights, indices

        if use_cache and policy == "unsafe_static":
            if self._cached_operator_weights is not None and self._cached_operator_indices is not None:
                return self._cached_operator_weights, self._cached_operator_indices

            source_mesh_beta = self.source_mesh_beta
            data_mesh_beta = self.data_mesh_beta
            weights, indices, _ = get_interpolation_weights(
                points=source_mesh_beta,
                query_points=data_mesh_beta,
                k_neighbors=self.pix_src_model.k_neighbors,
                kernel=self.pix_src_model.interp_kernel,
                radius_scale=self.pix_src_model.radius_scale,
            )
            self._cached_operator_weights = weights
            self._cached_operator_indices = indices
            self._cached_operator_signature = "unsafe_static"
            return weights, indices

        source_mesh_beta = self.source_mesh_beta
        data_mesh_beta = self.data_mesh_beta

        if use_cache and policy == "safe":
            signature = self._operator_signature_from(source_mesh_beta, data_mesh_beta)
            if (
                self._cached_operator_weights is not None
                and self._cached_operator_indices is not None
                and self._cached_operator_signature == signature
            ):
                return self._cached_operator_weights, self._cached_operator_indices

            weights, indices, _ = get_interpolation_weights(
                points=source_mesh_beta,
                query_points=data_mesh_beta,
                k_neighbors=self.pix_src_model.k_neighbors,
                kernel=self.pix_src_model.interp_kernel,
                radius_scale=self.pix_src_model.radius_scale,
            )
            self._cached_operator_weights = weights
            self._cached_operator_indices = indices
            self._cached_operator_signature = signature
            return weights, indices

        weights, indices, _ = get_interpolation_weights(
            points=source_mesh_beta,
            query_points=data_mesh_beta,
            k_neighbors=self.pix_src_model.k_neighbors,
            kernel=self.pix_src_model.interp_kernel,
            radius_scale=self.pix_src_model.radius_scale,
        )
        return weights, indices
    
    def build_blurred_lens_mapping_matrix(self, method: str = 'fft') -> jnp.ndarray:
        return apply_psf_to_mapping_matrix(
            mapping_matrix=self.build_lens_mapping_matrix(),
            psf_kernel=self.psf_kernel,
            image_shape=self.image_shape,
            unmasked_indices=self.unmasked_indices,
            method=method,
            psf_fft=self.psf_fft if method == 'fft' else None,
            psf_shape=self.psf_shape if method == 'fft' else None,
        )

    def build_lens_light_basis_matrix(self, method: str = 'fft') -> jnp.ndarray:
        n_lens_light = len(self.phys_model.lens_light)
        if n_lens_light == 0:
            return jnp.zeros((self.xgrid_unmask.shape[0], 0), dtype=jnp.float32)

        basis = []
        for light_model in self.phys_model.lens_light:
            basis.append(light_model.light(x=self.xgrid_unmask, y=self.ygrid_unmask))
        basis_unblurred = jnp.stack(basis, axis=1).astype(jnp.float32)

        return apply_psf_to_mapping_matrix(
            mapping_matrix=basis_unblurred,
            psf_kernel=self.psf_kernel,
            image_shape=self.image_shape,
            unmasked_indices=self.unmasked_indices,
            method=method,
            psf_fft=self.psf_fft if method == 'fft' else None,
            psf_shape=self.psf_shape if method == 'fft' else None,
        )
    
    def build_regularization_matrix(self, reg_scale: float, reg_coefficient: float) -> jnp.ndarray:
        if getattr(self.pix_src_model, "is_rectangular_grid", False):
            raise ValueError(
                "build_regularization_matrix() is unavailable for rectangular source grid; "
                "matrix-mode rectangular regularization is assembled inside build_inverter()."
            )
        return self.pix_src_model.regularization_matrix(
            points=self.source_mesh_beta,
            reg_scale=reg_scale,
            reg_coefficient=reg_coefficient,
        )

    def build_inverter(
        self,
        data_vector: Union[jnp.ndarray, np.ndarray],
        noise_variance: Union[jnp.ndarray, np.ndarray],
        reg_scale: float,
        reg_coefficient: float,
        *,
        include_lens_light: bool = False,
        lens_light_ridge: float = 1e-8,
        nonnegative: bool = False,
        inversion_backend: str = "matrix",
        cg_tol: float = 1e-4,
        cg_maxiter: int = 120,
        slq_seed: int = 0,
        slq_probes: int = 32,
        slq_steps: int = 60,
        evidence_mode: str = "accurate",
        operator_cache_policy: str = "safe",
        nnls_maxiter: int = 600,
        nnls_tol: float = 1e-6,
        nnls_lipschitz_iters: int = 12,
    ) -> Union[LinearInversion, OperatorInversion, NNLSInversion, OperatorNNLSInversion]:
        """Build and configure the semi-linear inversion solver.

        This method is the central backend dispatcher for source reconstruction.
        It supports both dense matrix inversion and matrix-free operator inversion,
        while keeping the numerical interface consistent across source-grid types.

        Notes
        -----
        Backend behavior by source-grid type:

        - ``source_grid_type='irregular'``:
          - ``matrix`` backend uses dense mapping and dense regularization.
          - ``operator`` backend uses matrix-free mapping with optional sparse
            regularization operators.
        - ``source_grid_type='rectangular_bilinear'``:
          - Mapping may be built either as a dense matrix (matrix backend) or as
            sparse bilinear operator entries (operator backend).
          - Regularization is first constructed in sparse COO form from the
            rectangular stencil and then either:
            - consumed directly by operator backends, or
            - densified for matrix backends.

        The sparse-first regularization pathway guarantees that rectangular
        regularization semantics are shared across both backends.
        """
        d = jnp.array(data_vector)
        noise_var = jnp.array(noise_variance)
        backend = _normalize_inversion_backend(inversion_backend)

        if getattr(self.pix_src_model, "is_rectangular_grid", False):
            reg_operator_mode = "sparse_rectangular"
        else:
            reg_operator_mode = str(getattr(self.pix_src_model, "reg_operator_mode", "dense_gp")).strip().lower()
        sparse_reg_enabled = reg_operator_mode in {"sparse_knn", "sparse_rectangular"}
        sparse_rows = sparse_cols = sparse_values = None
        sparse_n_source = None

        if reg_operator_mode == "sparse_knn":
            from TinyLensGpu.utils.lensing import regularization_sparse_knn_from

            sparse_rows, sparse_cols, sparse_values, sparse_n_source = regularization_sparse_knn_from(
                scale=float(reg_scale),
                coefficient=float(reg_coefficient),
                points=self.source_mesh_beta,
                reg_type=self.pix_src_model.reg_type,
                k_neighbors=int(getattr(self.pix_src_model, "reg_sparse_k_neighbors", 16)),
            )

        if reg_operator_mode == "sparse_rectangular":
            if self.source_grid_shape is None:
                raise RuntimeError("Rectangular source grid metadata missing for regularization.")
            ny, nx = self.source_grid_shape
            # Build rectangular regularization in sparse COO form once and reuse
            # it in both backends. Operator backends consume COO entries directly,
            # while matrix backends densify the same operator for exact solves.
            sparse_rows, sparse_cols, sparse_values, sparse_n_source = regularization_sparse_rectangular_from(
                coefficient=float(reg_coefficient),
                nx=int(nx),
                ny=int(ny),
                reg_scheme=str(getattr(self.pix_src_model, "rect_reg_type", "gradient")),
            )

        if backend == "operator":
            if sparse_reg_enabled:
                # Sparse operator mode does not require storing dense H.
                reg_matrix_src = jnp.zeros((int(self.source_mesh_beta.shape[0]),), dtype=jnp.float32)
            else:
                reg_matrix_src = self.pix_src_model.regularization_matrix(
                    points=self.source_mesh_beta,
                    reg_scale=reg_scale,
                    reg_coefficient=reg_coefficient,
                )
            lens_basis_matrix = None
            if include_lens_light:
                lens_basis_matrix = self.build_lens_light_basis_matrix()
            weights, indices = self.build_lens_mapping_operator(
                use_cache=True,
                cache_policy=operator_cache_policy,
            )
            if nonnegative:
                return OperatorNNLSInversion(
                    d=d,
                    noise_var=noise_var,
                    H=reg_matrix_src,
                    weights=weights,
                    indices=indices,
                    psf_fft=self.psf_fft,
                    image_shape=self.image_shape,
                    psf_shape=self.psf_shape,
                    unmasked_indices=self.unmasked_indices,
                    maxiter=nnls_maxiter,
                    tol=nnls_tol,
                    lipschitz_iters=nnls_lipschitz_iters,
                    fista_seed=slq_seed,
                    evidence_mode=evidence_mode,
                    slq_seed=slq_seed,
                    slq_probes=slq_probes,
                    slq_steps=slq_steps,
                    reg_operator_mode=reg_operator_mode,
                    lens_basis=lens_basis_matrix,
                    lens_light_ridge=lens_light_ridge,
                    H_sparse_rows=sparse_rows,
                    H_sparse_cols=sparse_cols,
                    H_sparse_values=sparse_values,
                    H_sparse_n_source=sparse_n_source,
                )
            return OperatorInversion(
                d=d,
                noise_var=noise_var,
                H=reg_matrix_src,
                weights=weights,
                indices=indices,
                psf_fft=self.psf_fft,
                image_shape=self.image_shape,
                psf_shape=self.psf_shape,
                unmasked_indices=self.unmasked_indices,
                cg_tol=cg_tol,
                cg_maxiter=cg_maxiter,
                slq_seed=slq_seed,
                slq_probes=slq_probes,
                slq_steps=slq_steps,
                evidence_mode=evidence_mode,
                reg_operator_mode=reg_operator_mode,
                lens_basis=lens_basis_matrix,
                lens_light_ridge=lens_light_ridge,
                H_sparse_rows=sparse_rows,
                H_sparse_cols=sparse_cols,
                H_sparse_values=sparse_values,
                H_sparse_n_source=sparse_n_source,
            )

        if backend == "matrix":
            if sparse_reg_enabled:
                if sparse_rows is None or sparse_cols is None or sparse_values is None or sparse_n_source is None:
                    raise RuntimeError("Sparse regularization mode selected but sparse entries are missing.")
                # Convert COO sparse regularization into dense H for exact matrix
                # backend solvers. This keeps the matrix backend numerically
                # aligned with the operator backend while still enabling direct
                # linear-system solves and block regularization with lens light.
                reg_matrix_src = sparse_regularization_dense_from(
                    rows=sparse_rows,
                    cols=sparse_cols,
                    values=sparse_values,
                    n_source=sparse_n_source,
                )
            else:
                reg_matrix_src = self.pix_src_model.regularization_matrix(
                    points=self.source_mesh_beta,
                    reg_scale=reg_scale,
                    reg_coefficient=reg_coefficient,
                )

        blurred_lens_map_matrix = self.build_blurred_lens_mapping_matrix()
        if not include_lens_light:
            if nonnegative:
                return NNLSInversion(d=d, F=blurred_lens_map_matrix, noise_cov=noise_var, H=reg_matrix_src)
            return LinearInversion(d=d, F=blurred_lens_map_matrix, noise_cov=noise_var, H=reg_matrix_src)

        lens_basis_matrix = self.build_lens_light_basis_matrix()
        F_total = jnp.concatenate([blurred_lens_map_matrix, lens_basis_matrix], axis=1)

        n_src = int(reg_matrix_src.shape[0])
        n_lens = int(lens_basis_matrix.shape[1])
        ridge = jnp.array(max(float(lens_light_ridge), 0.0), dtype=jnp.float32)
        H_lens = ridge * jnp.eye(n_lens, dtype=jnp.float32)
        H_total = jnp.block(
            [
                [reg_matrix_src, jnp.zeros((n_src, n_lens), dtype=jnp.float32)],
                [jnp.zeros((n_lens, n_src), dtype=jnp.float32), H_lens],
            ]
        )

        if nonnegative:
            return NNLSInversion(d=d, F=F_total, noise_cov=noise_var, H=H_total)
        return LinearInversion(d=d, F=F_total, noise_cov=noise_var, H=H_total)

    def reconstruct_source_and_lens_light(
        self,
        data_vector: Union[jnp.ndarray, np.ndarray],
        noise_variance: Union[jnp.ndarray, np.ndarray],
        reg_scale: float,
        reg_coefficient: float,
        lens_light_ridge: float = 1e-8,
        nonnegative: bool = True,
        return_2d: bool = False,
        inversion_backend: str = "matrix",
        **kwargs,
    ):
        """Jointly reconstruct source and lens-light linear coefficients.

        This routine supports both ``inversion_backend='matrix'`` and
        ``inversion_backend='operator'`` for joint source+lens-light inference.

        Rectangular source grids are supported: their sparse regularization
        stencils are internally densified before assembling the source/lens-light
        block regularization matrix (matrix backend) or consumed as sparse
        operators (operator backend).
        """
        inverter = self.build_inverter(
            data_vector=data_vector,
            noise_variance=noise_variance,
            reg_scale=reg_scale,
            reg_coefficient=reg_coefficient,
            include_lens_light=True,
            lens_light_ridge=lens_light_ridge,
            nonnegative=nonnegative,
            inversion_backend=inversion_backend,
            **kwargs,
        )

        x_total = inverter.solve()
        n_src = int(self.source_mesh_beta.shape[0])
        source_intensities = x_total[:n_src]
        lens_light_intensities = x_total[n_src:]

        model_data = inverter.model_predict(x_total)

        if return_2d:
            model_image = jnp.zeros((self.npix, self.npix))
            model_image = model_image.at[~self.mask].set(model_data)
        else:
            model_image = model_data

        return source_intensities, lens_light_intensities, self.source_mesh_beta, model_image, inverter

    def reconstruct_source(
        self,
        data_vector: Union[jnp.ndarray, np.ndarray],
        noise_variance: Union[jnp.ndarray, np.ndarray],
        reg_scale: float,
        reg_coefficient: float,
        return_2d: bool = False,
        **kwargs,
    ) -> Tuple[
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
        Union[LinearInversion, OperatorInversion, NNLSInversion, OperatorNNLSInversion],
    ]:
        """
        Reconstruct the source given observed data and noise.

        This method performs the full source reconstruction pipeline:
        1. Build lensing mapping matrix with PSF convolution
        2. Build regularization matrix
        3. Solve the linear inverse problem for source intensities
        4. Generate model image

        Parameters
        ----------
        data_vector : jnp.ndarray or np.ndarray
            Observed image data vector (unmasked pixels only)
        noise_variance : jnp.ndarray or np.ndarray
            Noise variance for each unmasked pixel
        reg_scale : float
            Regularization scale parameter
        reg_coefficient : float
            Regularization coefficient (lambda)
        return_2d : bool, optional
            If True, returns the model image as a 2D array.
            If False (default), returns the model image as a 1D vector (unmasked pixels only).
        **kwargs : dict
            Additional arguments passed to build_inverter (e.g. inversion_backend, cg_tol, etc.)

        Returns
        -------
        source_intensities : jnp.ndarray
            Reconstructed source intensities at source mesh points
        source_mesh_beta : jnp.ndarray
            Source mesh coordinates in source plane (shape: n_source x 2)
        model_image : jnp.ndarray
            Model image. Shape is (npix, npix) if return_2d=True, 
            else (n_unmasked_pixels,) if return_2d=False.
        inverter : Union[LinearInversion, OperatorInversion, NNLSInversion, OperatorNNLSInversion]
            Linear inversion solver object (cached for reuse)
        """
        inverter = self.build_inverter(
            data_vector=data_vector,
            noise_variance=noise_variance,
            reg_scale=reg_scale,
            reg_coefficient=reg_coefficient,
            **kwargs,
        )

        # Solve for source intensities
        source_intensities = inverter.solve()

        # Generate model data
        model_data = inverter.model_predict(source_intensities)

        if return_2d:
            # Place model data into full image
            model_image = jnp.zeros((self.npix, self.npix))
            model_image = model_image.at[~self.mask].set(model_data)
        else:
            model_image = model_data

        return source_intensities, self.source_mesh_beta, model_image, inverter

    def __repr__(self) -> str:
        return (f"PixelizedLensSimulator("
                f"npix={self.npix}, "
                f"n_source_points={self.pix_src_model.n_source_points}, "
                f"n_mass={len(self.phys_model.lens_mass)})")
