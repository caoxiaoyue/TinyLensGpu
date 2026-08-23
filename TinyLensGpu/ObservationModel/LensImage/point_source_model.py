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
from jax import Array, jit, lax

from TinyLensGpu.Inference.param_u import ParamU
from TinyLensGpu.PhysicalModel.LensImage.composite import PhysicalModel
from TinyLensGpu.utils.lensing.point_source_solver import (
    min_assignment_chi2_hungarian,
    min_assignment_chi2_subset,
    select_unique_images_and_dists_fixed,
    solve_lens_equation_mesh_refine_core,
    solve_lens_equation_optimization_core,
)


class PointSourceProbModel(ck.Module):
    """
    Likelihood model for multiply imaged point-source positions.

    Given observed image positions and astrometric uncertainties, this model
    solves the lens equation for a trial source position and evaluates a
    permutation-invariant Gaussian position likelihood.

    Parameters
    ----------
    phys_model : PhysicalModel
        Lens mass model used for ray tracing.
    observed_positions : array_like
        Observed image positions with shape ``(n_images, 2)``.
    position_sigma : array_like or float
        Per-image 1-sigma astrometric uncertainty.
    source_x, source_y : ParamU or float, optional
        Source-plane coordinates.
    source_position_fixed : bool, optional
        If ``True``, keep source coordinates static in the caskade graph.
    solver : {'optimization', 'amr'}, optional
        Lens-equation solver backend.
    solver_config : dict, optional
        Numerical settings for the selected solver.
    matching : {'global_min_cost'}, optional
        Predicted/observed image matching strategy.
    min_log_like : float, optional
        Likelihood floor used for invalid solutions.
    """

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
        """
        Initialize a `PointSourceProbModel`.

        This model computes the likelihood of observing point sources (e.g., quasar images)
        at specific positions, given a lens mass model and a source position.

        Parameters
        ----------
        phys_model : PhysicalModel
            The physical model definition containing the lens mass profile.
        observed_positions : Union[Array, np.ndarray, Sequence[Sequence[float]]]
            A collection of (x, y) coordinates of the observed point source images.
            Shape: (N_images, 2).
        position_sigma : Union[Array, np.ndarray, Sequence[float], float]
            The astrometric uncertainty (1-sigma) of the observed positions.
            Can be a scalar or an array of shape (N_images,).
        source_x : Optional[Union[ParamU, float]], optional
            The source x-coordinate. Can be a fixed float or a `ParamU` parameter.
            Defaults to None (creates a default parameter).
        source_y : Optional[Union[ParamU, float]], optional
            The source y-coordinate. Can be a fixed float or a `ParamU` parameter.
            Defaults to None (creates a default parameter).
        source_position_fixed : bool, optional
            If True, the source position parameters are treated as static (non-optimizable)
            values in the `caskade` graph. Defaults to False.
        solver : str, optional
            The algorithm used to solve the lens equation. Options:
            - 'optimization': Uses gradient-based optimization to find image positions.
            - 'amr': Uses Adaptive Mesh Refinement to find image positions.
            Defaults to "optimization".
        solver_config : Optional[Dict], optional
            Configuration dictionary for the chosen solver (e.g., number of iterations, grid depth).
        matching : str, optional
            Strategy to match predicted image positions to observed positions.
            Currently only 'global_min_cost' (Hungarian algorithm or permutation) is supported.
        min_log_like : float, optional
            Minimum log-likelihood floor to avoid numerical issues when no images are found.
            Defaults to -1.0e12.

        Raises
        ------
        ValueError
            If `observed_positions` shape is invalid or `position_sigma` is non-positive.
        """
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

        # The pure-JAX subset matcher has O(K * N * 2**N) complexity. It is
        # efficient for normal double/quad systems; larger multiplicities keep
        # the existing SciPy Hungarian fallback for compatibility.
        object.__setattr__(self, "_use_subset_dp", self.n_observed <= 8)

        # Precompute log-normalization constant for Gaussian likelihood: sum(log(2 * pi * sigma^2))
        # This corresponds to the constant term in the 2D Gaussian log-likelihood for N points.
        log_norm = jnp.sum(jnp.log(2.0 * jnp.pi * jnp.square(sigma)))
        object.__setattr__(self, "_log_norm", log_norm)

    @staticmethod
    def _build_source_param(name: str, value: Optional[Union[ParamU, float]]) -> ParamU:
        """
        Helper to construct a source position parameter.

        Ensures the input is wrapped in a `ParamU` object with appropriate defaults
        if not already provided.

        Parameters
        ----------
        name : str
            Name of the parameter (e.g., "source_x").
        value : Optional[Union[ParamU, float]]
            The input value or parameter object.

        Returns
        -------
        ParamU
            The constructed parameter object.
        """
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
        """
        Retrieve the dynamic parameters of this model.

        Returns
        -------
        dict
            Dictionary of dynamic parameters (e.g., source position if not fixed).
        """
        return self.dynamic_params

    @ck.forward
    def _ray_trace(self, theta: Array) -> Array:
        """
        Map image plane coordinates to the source plane.

        Parameters
        ----------
        theta : Array
            Image plane coordinates of shape (..., 2), where the last dimension contains (x, y).

        Returns
        -------
        beta : Array
            Source plane coordinates of shape (..., 2).
        """
        x = theta[..., 0]
        y = theta[..., 1]
        beta_x, beta_y = self.phys_model.deflection(x, y)
        return jnp.stack([beta_x, beta_y], axis=-1)

    @ck.forward
    @functools.partial(jit, static_argnums=(0,))
    def solve_image_positions_fixed(self) -> Tuple[Array, Array, Array, Array]:
        """Solve image positions with a static output shape for JIT reuse.

        Returns
        -------
        images, dists, valid_mask, count
            Padded arrays of shape ``(k_keep, 2)`` and ``(k_keep,)``, a mask
            for valid roots, and their count. Invalid rows are zero padded.
        """
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

        return select_unique_images_and_dists_fixed(
            images=candidates,
            dists=dists,
            n_select=self._cfg_k_keep,
            tolerance=self._cfg_tolerance,
            cluster_tol=self._cfg_cluster_tol,
        )

    @ck.forward
    def solve_image_positions(self) -> Tuple[Array, Array]:
        """
        Solve the lens equation to find image positions for the current source position.

        Uses the configured solver (Optimization or AMR) to find roots of the lens equation.

        Returns
        -------
        candidates : Array
            The found image positions (x, y).
        dists : Array
            The distance in the source plane between the ray-traced image position and the true source position.
            Used to filter valid solutions.
        """
        images, dists, _, count = self.solve_image_positions_fixed()
        valid_count = int(np.asarray(count))
        return images[:valid_count], dists[:valid_count]

    @ck.forward
    @functools.partial(jit, static_argnums=(0,))
    def __call__(self) -> Array:
        """
        Compute the log-likelihood of the observed point source positions.

        1. Solves the lens equation to find predicted image positions.
        2. Filters valid images based on the source plane distance threshold.
        3. Matches observed images one-to-one to the best subset of all valid
           predicted images, without penalizing unobserved extra roots.
        4. Computes the Gaussian log-likelihood.

        Returns
        -------
        log_like : Array
            The log-likelihood value. Returns `min_log_like` if there are
            fewer valid predicted images than observed positions.
        """
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

        selected, _, selected_mask, count = select_unique_images_and_dists_fixed(
            images=candidates,
            dists=dists,
            n_select=self._cfg_k_keep,
            tolerance=self._cfg_tolerance,
            cluster_tol=self._cfg_cluster_tol,
        )

        finite_ok = jnp.logical_and(
            jnp.all(jnp.isfinite(candidates)),
            jnp.all(jnp.isfinite(dists)),
        )
        enough_images = count >= self.n_observed
        valid = jnp.logical_and(finite_ok, enough_images)

        if self._use_subset_dp:
            chi2 = min_assignment_chi2_subset(
                observed_positions=self.observed_positions,
                predicted_positions=selected,
                sigma_pos=self.position_sigma,
                predicted_valid=selected_mask,
            )
        else:
            chi2 = lax.cond(
                enough_images,
                lambda _: min_assignment_chi2_hungarian(
                    observed_positions=self.observed_positions,
                    predicted_positions=selected,
                    sigma_pos=self.position_sigma,
                    predicted_valid=selected_mask,
                ),
                lambda _: jnp.asarray(jnp.inf, dtype=jnp.float32),
                operand=None,
            )
        log_like = -0.5 * chi2 - self._log_norm

        return jnp.where(valid, log_like, self.min_log_like)

    def likelihood(self) -> float:
        """
        Compute the scalar likelihood value (non-JIT wrapper).

        Returns
        -------
        float
            The log-likelihood value.
        """
        return float(np.asarray(self.__call__()))

    def __repr__(self) -> str:
        """
        Return a string representation of the `PointSourceProbModel`.

        Returns
        -------
        str
            A string summarizing the model state (observed count, solver type, matching strategy).
        """
        return (
            f"PointSourceProbModel(n_observed={self.n_observed}, solver='{self.solver}', "
            f"matching='{self.matching}')"
        )
