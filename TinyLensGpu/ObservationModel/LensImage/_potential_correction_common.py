"""Shared numerical preparation for source–potential reconstruction."""

from inspect import signature

import jax
import jax.numpy as jnp
import numpy as np
from TinyLensGpu.ForwardSimulation import PotentialCorrectionRenderer
from TinyLensGpu.utils.inversion.potential_regularization import PotentialRegularization
from TinyLensGpu.utils.inversion.regularization import FiniteDifferenceRegularization
from .potential_correction_result import PotentialCorrectionStrengths


def _dense_response_columns(action, size, dtype):
    """Materialize a dense response with bounded subpixel/image temporaries.

    Only sixteen basis responses are evaluated simultaneously. Generating each
    basis vector inside the map avoids an additional complete identity array.
    The returned dense design is intentional for the reference backend.
    """

    def column(index):
        return action(jax.nn.one_hot(index, size, dtype=dtype))

    indices = jnp.arange(size, dtype=jnp.int32)
    if "batch_size" in signature(jax.lax.map).parameters:
        return jax.lax.map(column, indices, batch_size=16).T

    # JAX 0.4.20 exposes map(f, xs). Map whole bounded panels there, with
    # a separate remainder, preserving the modern path's order and memory bound.
    full_columns = (size // 16) * 16
    parts = []
    if full_columns:
        panels = jax.lax.map(jax.vmap(column), indices[:full_columns].reshape((-1, 16)))
        parts.append(panels.reshape((full_columns,) + panels.shape[2:]))
    if full_columns < size:
        parts.append(jax.vmap(column)(indices[full_columns:]))
    columns = parts[0] if len(parts) == 1 else jnp.concatenate(parts, axis=0)
    return columns.T


def _positive(value, name):
    try:
        scalar = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite positive scalar") from exc
    if not np.isfinite(scalar) or scalar <= 0:
        raise ValueError(f"{name} must be a finite positive scalar")
    return scalar


class _PotentialCorrectionProblem:
    """Observation, fixed geometry, physical priors and raw result products."""

    def __init__(
        self,
        renderer,
        image_data,
        noise_map,
        *,
        lambda_amplitude,
        source_regularization="second-order",
        source_scale=None,
        macro_mapping=None,
        solve_rtol=None,
        solve_atol=0.0,
    ):
        if not isinstance(renderer, PotentialCorrectionRenderer):
            raise TypeError("renderer must be a PotentialCorrectionRenderer")
        self.renderer = renderer
        self.dtype = renderer.dtype
        self.lambda_amplitude = _positive(lambda_amplitude, "lambda_amplitude")
        self.solve_rtol = _positive(
            (
                (1e-8 if self.dtype == jnp.float64 else 1e-3)
                if solve_rtol is None
                else solve_rtol
            ),
            "solve_rtol",
        )
        if not np.isfinite(solve_atol) or solve_atol < 0:
            raise ValueError("solve_atol must be finite and nonnegative")
        self.solve_atol = float(solve_atol)
        self.n_source = renderer.n_source
        self.n_potential = renderer.potential_grid.n_unknown
        self.n_joint = self.n_source + self.n_potential
        self.source_regularizer = FiniteDifferenceRegularization(
            renderer.source_n, source_regularization
        )
        self.source_scale = None
        if source_scale is not None:
            scale = np.array(source_scale, copy=True)
            if scale.shape != (self.n_source,) or not np.all(
                np.isfinite(scale) & (scale > 0)
            ):
                raise ValueError(
                    "source_scale must contain n_source finite positive values"
                )
            with jax.experimental.enable_x64(self.dtype == jnp.float64):
                self.source_scale = jnp.asarray(scale, dtype=self.dtype)
        self.potential_regularizer = PotentialRegularization(renderer.potential_grid)
        self.macro_mapping = (
            renderer.macro_mapping if macro_mapping is None else macro_mapping
        )
        with jax.experimental.enable_x64(self.dtype == jnp.float64):
            self.image_data = jnp.asarray(image_data, dtype=self.dtype)
            self.noise_map = jnp.asarray(noise_map, dtype=self.dtype)
            if (
                self.image_data.shape != renderer.image_shape
                or self.noise_map.shape != renderer.image_shape
            ):
                raise ValueError(
                    "image_data and noise_map must match the native image shape"
                )
            self.data_active = self.image_data.ravel()[renderer.flat_indices]
            self.noise_active = self.noise_map.ravel()[renderer.flat_indices]
            if not bool(jnp.all(jnp.isfinite(self.data_active))):
                raise ValueError("Selected image_data must be finite")
            if not bool(
                jnp.all(jnp.isfinite(self.noise_active) & (self.noise_active > 0))
            ):
                raise ValueError("Selected noise_map must be finite and positive")
            self.noise_logdet = 2 * jnp.sum(jnp.log(self.noise_active))
        self._dense_source_prior = None
        self._dense_potential_roughness = None

    def _joint(self, joint_values):
        values = jnp.asarray(joint_values, dtype=self.dtype)
        if values.shape != (self.n_joint,):
            raise ValueError(f"joint_values must have shape ({self.n_joint},)")
        return values[: self.n_source], values[self.n_source :]

    def prior_matvec(self, joint_values, lambda_source, lambda_potential):
        """Apply the complete block prior without creating square matrices."""
        with jax.experimental.enable_x64(self.dtype == jnp.float64):
            source, potential = self._joint(joint_values)
            return jnp.concatenate(
                (
                    jnp.asarray(lambda_source, dtype=self.dtype)
                    * self.source_regularizer.matvec_free(
                        source, *self.renderer.source_bbox, scale=self.source_scale
                    ),
                    self.potential_regularizer.matvec(
                        potential,
                        lambda_potential,
                        self.lambda_amplitude,
                    ),
                )
            )

    def dense_prior(self, lambda_source, lambda_potential):
        """Explicitly materialize Q, including the independent all-node prior."""
        with jax.experimental.enable_x64(self.dtype == jnp.float64):
            if self._dense_source_prior is None:
                self._dense_source_prior = self.source_regularizer.matrix(
                    *self.renderer.source_bbox,
                    scale=self.source_scale,
                ).astype(self.dtype)
                self._dense_potential_roughness = self.potential_regularizer.to_dense(
                    1, 0
                )
            prior = jnp.zeros((self.n_joint, self.n_joint), dtype=self.dtype)
            prior = prior.at[: self.n_source, : self.n_source].set(
                jnp.asarray(lambda_source, dtype=self.dtype) * self._dense_source_prior
            )
            return prior.at[self.n_source :, self.n_source :].set(
                jnp.asarray(lambda_potential, dtype=self.dtype)
                * self._dense_potential_roughness
                + jnp.asarray(self.lambda_amplitude, dtype=self.dtype)
                * jnp.eye(self.n_potential, dtype=self.dtype)
            )

    def _configuration(self):
        return {
            "source_n": self.renderer.source_n,
            "source_interpolation": self.renderer.source_interpolation,
            "source_boundary": self.renderer.source_boundary,
            "gradient_policy": self.initialization_provenance["gradient_policy"],
            "source_regularization": self.source_regularizer.regularization_type,
            "source_adaptive_scale_fixed": self.source_scale is not None,
            "potential_spacing": self.renderer.potential_grid.spacing,
            "potential_regularization": "complete_pixel_unit_fourth_differences",
            "potential_amplitude_prior": "independent_all_unknowns",
            "nsub": self.renderer.nsub,
            "image_shape": self.renderer.image_shape,
            "n_data": self.renderer.n_active,
            "n_source": self.n_source,
            "n_potential": self.n_potential,
            "n_core": self.renderer.potential_grid.n_core,
            "solve_rtol": self.solve_rtol,
            "dtype": self.dtype.name,
        }

    def _result_products(
        self, joint_values, lambda_source, lambda_potential, *, linear_image=None
    ):
        source, potential = self._joint(joint_values)
        finite = bool(jnp.all(jnp.isfinite(source)) & jnp.all(jnp.isfinite(potential)))
        if finite:
            rendered = self.renderer.render(
                source, potential, macro_mapping=self.macro_mapping
            )
            image, active = rendered.model_image, rendered.image_active
            ax, ay, kappa = (
                rendered.deflection_x,
                rendered.deflection_y,
                rendered.convergence,
            )
        else:
            image = jnp.full(self.renderer.image_shape, jnp.nan, dtype=self.dtype)
            active = jnp.full_like(self.data_active, jnp.nan)
            ax = ay = kappa = jnp.full_like(potential, jnp.nan)
        nonlinear_residual = self.data_active - active
        linear_active = (
            None
            if linear_image is None
            else linear_image.ravel()[self.renderer.flat_indices]
        )
        linear_residual = (
            None if linear_active is None else self.data_active - linear_active
        )
        residual = nonlinear_residual if linear_residual is None else linear_residual
        chi2 = jnp.vdot(residual / self.noise_active, residual / self.noise_active)
        penalty = jnp.vdot(
            jnp.asarray(joint_values, dtype=self.dtype),
            self.prior_matvec(joint_values, lambda_source, lambda_potential),
        )

        def full(values):
            if values is None:
                return None
            return (
                jnp.full(
                    self.renderer.image_shape[0] * self.renderer.image_shape[1],
                    jnp.nan,
                    dtype=self.dtype,
                )
                .at[self.renderer.flat_indices]
                .set(values)
                .reshape(self.renderer.image_shape)
            )

        return dict(
            source_values=source,
            source_x_axis=self.renderer.source_x_axis,
            source_y_axis=self.renderer.source_y_axis,
            source_bbox=self.renderer.source_bbox,
            potential_values=potential,
            potential_grid=self.renderer.potential_grid,
            deflection_x=ax,
            deflection_y=ay,
            convergence=kappa,
            derivative_valid=self.renderer.potential_grid.derivative_valid,
            potential_window_coverage=self.potential_regularizer.window_coverage,
            linear_model_image=linear_image,
            nonlinear_model_image=image,
            linear_image_active=linear_active,
            nonlinear_image_active=active,
            linear_residual=full(linear_residual),
            nonlinear_residual=full(nonlinear_residual),
            linear_normalized_residual=full(
                None if linear_residual is None else linear_residual / self.noise_active
            ),
            nonlinear_normalized_residual=full(nonlinear_residual / self.noise_active),
            data_active=self.data_active,
            noise_active=self.noise_active,
            strengths=PotentialCorrectionStrengths(
                lambda_source, lambda_potential, self.lambda_amplitude
            ),
            chi2=chi2,
            regularization_penalty=penalty,
            objective=0.5 * (chi2 + penalty),
            noise_logdet=self.noise_logdet,
            dtype=self.dtype.name,
        )
