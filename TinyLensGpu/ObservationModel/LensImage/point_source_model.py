"""
Point-source position likelihood model for strong lensing.

This module provides a standalone likelihood model based on observed lensed
image positions of a point source.
"""

from __future__ import annotations

import functools
from typing import Dict, Optional, Sequence, Tuple, Union

import caskade as ck
import jax.numpy as jnp
import numpy as np
from jax import Array, jit

from TinyLensGpu.Inference.param_u import ParamU
from TinyLensGpu.PhysicalModel.LensImage.composite import PhysicalModel
from TinyLensGpu.utils.lensing.point_source_solver import (
    build_permutation_indices,
    min_assignment_chi2,
    min_assignment_chi2_hungarian,
    select_unique_images_fixed,
    solve_lens_equation_mesh_refine,
    solve_lens_equation_mesh_refine_core,
    solve_lens_equation_optimization,
    solve_lens_equation_optimization_core,
)


class PointSourceProbModel(ck.Module):
    """Likelihood model using lensed point-image positions."""

    def __init__(
        self,
        phys_model: PhysicalModel,
        observed_positions: Union[Array, np.ndarray, Sequence[Sequence[float]]],
        position_sigma: Union[Array, np.ndarray, Sequence[float], float],
        source_x: Optional[Union[ParamU, float]] = None,
        source_y: Optional[Union[ParamU, float]] = None,
        source_position_fixed: bool = False,
        solver: str = "optimization",
        solver_config: Optional[Dict] = None,
        matching: str = "global_min_cost",
        min_log_like: float = -1.0e12,
    ) -> None:
        super().__init__("point_source_prob_model")

        self.phys_model = phys_model

        obs = jnp.asarray(observed_positions, dtype=jnp.float32)
        if obs.ndim != 2 or obs.shape[1] != 2:
            raise ValueError("observed_positions must have shape (N, 2)")
        if obs.shape[0] < 1:
            raise ValueError("observed_positions must contain at least one image")

        sigma = jnp.asarray(position_sigma, dtype=jnp.float32)
        if sigma.ndim == 0:
            sigma = jnp.full((obs.shape[0],), sigma, dtype=jnp.float32)
        if sigma.ndim != 1 or sigma.shape[0] != obs.shape[0]:
            raise ValueError("position_sigma must have shape (N,) matching observed_positions")
        if bool(jnp.any(sigma <= 0.0)):
            raise ValueError("position_sigma values must be positive")

        object.__setattr__(self, "observed_positions", obs)
        object.__setattr__(self, "position_sigma", sigma)
        object.__setattr__(self, "n_observed", int(obs.shape[0]))

        if matching != "global_min_cost":
            raise ValueError("Only matching='global_min_cost' is supported")
        object.__setattr__(self, "matching", matching)
        object.__setattr__(self, "min_log_like", jnp.asarray(float(min_log_like), dtype=jnp.float32))

        self.source_x = self._build_source_param("source_x", source_x)
        self.source_y = self._build_source_param("source_y", source_y)
        if source_position_fixed:
            self.source_x.to_static()
            self.source_y.to_static()
        else:
            self.source_x.to_dynamic()
            self.source_y.to_dynamic()

        solver_name = str(solver).strip().lower()
        if solver_name not in ("optimization", "amr"):
            raise ValueError("solver must be one of {'optimization', 'amr'}")
        object.__setattr__(self, "solver", solver_name)
        object.__setattr__(self, "_use_amr", solver_name == "amr")

        cfg = dict(solver_config or {})
        object.__setattr__(self, "_cfg_initial_range", float(cfg.get("initial_range", 5.0)))
        object.__setattr__(self, "_cfg_n_x", int(cfg.get("n_x", 100)))
        object.__setattr__(self, "_cfg_n_y", int(cfg.get("n_y", 100)))
        object.__setattr__(self, "_cfg_k_keep", int(cfg.get("k_keep", 20)))
        object.__setattr__(self, "_cfg_tolerance", float(cfg.get("tolerance", 1.0e-4)))
        object.__setattr__(self, "_cfg_cluster_tol", float(cfg.get("cluster_tol", 0.05)))

        object.__setattr__(self, "_cfg_num_iters", int(cfg.get("num_iters", 20)))
        object.__setattr__(self, "_cfg_jacobian_eps", float(cfg.get("jacobian_eps", 1.0e-6)))

        object.__setattr__(self, "_cfg_subgrid_res", int(cfg.get("subgrid_res", 20)))
        object.__setattr__(self, "_cfg_depth", int(cfg.get("depth", 10)))
        object.__setattr__(self, "_cfg_search_factor", float(cfg.get("search_factor", 2.0)))

        # Use Hungarian algorithm if N > 4 to avoid combinatorial explosion
        # Otherwise use permutation indices for GPU efficiency
        self._use_hungarian = self.n_observed > 4
        
        if self._use_hungarian:
            object.__setattr__(self, "_perm_indices", None)
        else:
            perms = build_permutation_indices(self.n_observed)
            object.__setattr__(self, "_perm_indices", perms)

        # Precompute log-normalization constant for Gaussian likelihood: sum(log(2 * pi * sigma^2))
        # This corresponds to the constant term in the 2D Gaussian log-likelihood for N points.
        log_norm = jnp.sum(jnp.log(2.0 * jnp.pi * jnp.square(sigma)))
        object.__setattr__(self, "_log_norm", log_norm)

    @staticmethod
    def _build_source_param(name: str, value: Optional[Union[ParamU, float]]) -> ParamU:
        if isinstance(value, ParamU):
            return value
        if value is None:
            return ParamU(
                name,
                0.0,
                prior_type="uniform",
                prior_settings=[-2.5, 2.5],
                limits=[-10.0, 10.0],
            )
        return ParamU(
            name,
            float(value),
            prior_type="uniform",
            prior_settings=[-2.5, 2.5],
            limits=[-10.0, 10.0],
        )

    def get_dynamic_params(self):
        """Return dynamic parameters for prior extraction."""
        return self.dynamic_params

    @ck.forward
    def _ray_trace(self, theta: Array) -> Array:
        x = theta[..., 0]
        y = theta[..., 1]
        beta_x, beta_y = self.phys_model.deflection(x, y)
        return jnp.stack([beta_x, beta_y], axis=-1)

    @ck.forward
    def solve_image_positions(self) -> Tuple[Array, Array]:
        """Solve for candidate image positions for current parameters."""
        source_pos = jnp.asarray([self.source_x.value, self.source_y.value], dtype=jnp.float32)

        if self._use_amr:
            return solve_lens_equation_mesh_refine(
                source_pos=source_pos,
                ray_trace_fn=self._ray_trace,
                initial_range=self._cfg_initial_range,
                n_x=self._cfg_n_x,
                n_y=self._cfg_n_y,
                k_keep=self._cfg_k_keep,
                subgrid_res=self._cfg_subgrid_res,
                depth=self._cfg_depth,
                search_factor=self._cfg_search_factor,
                tolerance=self._cfg_tolerance,
                cluster_tol=self._cfg_cluster_tol,
            )

        return solve_lens_equation_optimization(
            source_pos=source_pos,
            ray_trace_fn=self._ray_trace,
            initial_range=self._cfg_initial_range,
            n_x=self._cfg_n_x,
            n_y=self._cfg_n_y,
            k_keep=self._cfg_k_keep,
            num_iters=self._cfg_num_iters,
            tolerance=self._cfg_tolerance,
            cluster_tol=self._cfg_cluster_tol,
            jacobian_eps=self._cfg_jacobian_eps,
        )

    @ck.forward
    @functools.partial(jit, static_argnums=(0,))
    def __call__(self) -> Array:
        source_pos = jnp.asarray([self.source_x.value, self.source_y.value], dtype=jnp.float32)

        if self._use_amr:
            candidates, dists = solve_lens_equation_mesh_refine_core(
                source_pos=source_pos,
                ray_trace_fn=self._ray_trace,
                initial_range=self._cfg_initial_range,
                n_x=self._cfg_n_x,
                n_y=self._cfg_n_y,
                k_keep=self._cfg_k_keep,
                subgrid_res=self._cfg_subgrid_res,
                depth=self._cfg_depth,
                search_factor=self._cfg_search_factor,
            )
        else:
            candidates, dists = solve_lens_equation_optimization_core(
                source_pos=source_pos,
                ray_trace_fn=self._ray_trace,
                initial_range=self._cfg_initial_range,
                n_x=self._cfg_n_x,
                n_y=self._cfg_n_y,
                k_keep=self._cfg_k_keep,
                num_iters=self._cfg_num_iters,
                jacobian_eps=self._cfg_jacobian_eps,
            )

        selected, selected_mask, count = select_unique_images_fixed(
            images=candidates,
            dists=dists,
            n_select=self.n_observed,
            tolerance=self._cfg_tolerance,
            cluster_tol=self._cfg_cluster_tol,
        )

        finite_ok = jnp.logical_and(
            jnp.all(jnp.isfinite(candidates)),
            jnp.all(jnp.isfinite(dists)),
        )
        enough_images = count == self.n_observed
        enough_mask = jnp.all(selected_mask)
        valid = jnp.logical_and(finite_ok, jnp.logical_and(enough_images, enough_mask))

        if self._use_hungarian:
            chi2 = min_assignment_chi2_hungarian(
                observed_positions=self.observed_positions,
                predicted_positions=selected,
                sigma_pos=self.position_sigma,
            )
        else:
            chi2 = min_assignment_chi2(
                observed_positions=self.observed_positions,
                predicted_positions=selected,
                sigma_pos=self.position_sigma,
                permutation_indices=self._perm_indices,
            )
        log_like = -0.5 * chi2 - self._log_norm

        return jnp.where(valid, log_like, self.min_log_like)

    def likelihood(self) -> float:
        """Return scalar log-likelihood value."""
        return float(np.asarray(self.__call__()))

    def __repr__(self) -> str:
        return (
            f"PointSourceProbModel(n_observed={self.n_observed}, solver='{self.solver}', "
            f"matching='{self.matching}')"
        )

