"""
Point-source lens equation solvers and matching utilities.

This module provides reusable utilities for:
- Solving for image-plane positions given a source position
- Filtering/declustering candidate image solutions
- Computing permutation-invariant position chi-square
"""

from __future__ import annotations

import itertools
from functools import partial
from typing import Callable, Optional, Tuple

import jax
import numpy as np
import scipy.optimize
from jax import jacfwd, lax, vmap
import jax.numpy as jnp


def _find_initial_candidates(
    source_pos: jnp.ndarray,
    ray_trace_fn: Callable[[jnp.ndarray], jnp.ndarray],
    initial_range: float,
    n_x: int,
    n_y: int,
    k_keep: int,
) -> Tuple[jnp.ndarray, float]:
    """
    Find promising initial image candidates on a coarse grid.

    Candidate points are selected by local-minimum filtering of source-plane
    residual distance and top-k ranking.

    Parameters
    ----------
    source_pos : jnp.ndarray
        Target source position ``(2,)``.
    ray_trace_fn : callable
        Ray-tracing function mapping ``theta -> beta``.
    initial_range : float
        Half-width of square search region in image-plane units.
    n_x, n_y : int
        Number of grid cells along x/y.
    k_keep : int
        Number of best coarse candidates to keep.

    Returns
    -------
    tuple[jnp.ndarray, float]
        Candidate image positions of shape ``(k_keep, 2)`` and coarse pixel width.
    """
    xs = jnp.linspace(-initial_range, initial_range, n_x + 1)
    ys = jnp.linspace(-initial_range, initial_range, n_y + 1)
    grid_x, grid_y = jnp.meshgrid(xs, ys)
    initial_theta = jnp.stack([grid_x.reshape(-1), grid_y.reshape(-1)], axis=-1)

    beta_0 = ray_trace_fn(initial_theta)
    dist_0 = jnp.linalg.norm(beta_0 - source_pos, axis=-1)

    dist_grid = dist_0.reshape(n_y + 1, n_x + 1)
    padded_dist = jnp.pad(dist_grid, 1, mode='constant', constant_values=jnp.inf)

    is_min = (
        (dist_grid <= padded_dist[0:-2, 1:-1])
        & (dist_grid <= padded_dist[2:, 1:-1])
        & (dist_grid <= padded_dist[1:-1, 0:-2])
        & (dist_grid <= padded_dist[1:-1, 2:])
        & (dist_grid <= padded_dist[0:-2, 0:-2])
        & (dist_grid <= padded_dist[0:-2, 2:])
        & (dist_grid <= padded_dist[2:, 0:-2])
        & (dist_grid <= padded_dist[2:, 2:])
    )

    is_min = is_min.at[0, :].set(False)
    is_min = is_min.at[-1, :].set(False)
    is_min = is_min.at[:, 0].set(False)
    is_min = is_min.at[:, -1].set(False)

    penalty = jnp.where(is_min.reshape(-1), 0.0, 1.0e10)
    scores = -(dist_0 + penalty)

    n_total = initial_theta.shape[0]
    k_keep = min(max(k_keep, 1), n_total)
    _, indices = lax.top_k(scores, k_keep)

    pixel_width = (2.0 * float(initial_range)) / float(max(n_x, 1))
    return initial_theta[indices], pixel_width


def solve_lens_equation_optimization_core(
    source_pos: jnp.ndarray,
    ray_trace_fn: Callable[[jnp.ndarray], jnp.ndarray],
    initial_range: float = 5.0,
    n_x: int = 100,
    n_y: int = 100,
    k_keep: int = 20,
    num_iters: int = 20,
    jacobian_eps: float = 1.0e-6,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Solve lens equation using Newton refinement from coarse seeds.

    Parameters
    ----------
    source_pos : jnp.ndarray
        Target source position ``(2,)``.
    ray_trace_fn : callable
        Ray-tracing function.
    initial_range : float, optional
        Initial search half-width.
    n_x, n_y : int, optional
        Coarse-grid resolution.
    k_keep : int, optional
        Number of coarse candidates to refine.
    num_iters : int, optional
        Newton refinement iterations per candidate.
    jacobian_eps : float, optional
        Diagonal stabilization added to Jacobian solve.

    Returns
    -------
    tuple[jnp.ndarray, jnp.ndarray]
        Refined image candidates and corresponding source-plane residual norms,
        both sorted by residual.
    """
    source_pos = jnp.asarray(source_pos, dtype=jnp.float32)
    best_candidates, _ = _find_initial_candidates(
        source_pos=source_pos,
        ray_trace_fn=ray_trace_fn,
        initial_range=initial_range,
        n_x=n_x,
        n_y=n_y,
        k_keep=k_keep,
    )

    eye2 = jnp.eye(2, dtype=jnp.float32)

    def refine_candidate(start_theta: jnp.ndarray) -> jnp.ndarray:
        """Refine one candidate via fixed-iteration Newton updates."""
        def body_fn(theta: jnp.ndarray, _: None) -> Tuple[jnp.ndarray, None]:
            """Single Newton step for a candidate coordinate."""
            def f(t: jnp.ndarray) -> jnp.ndarray:
                """Lens equation residual ``beta(theta) - source_pos``."""
                return ray_trace_fn(t) - source_pos

            val = f(theta)
            jac = jacfwd(f)(theta)
            delta = jnp.linalg.solve(jac + jacobian_eps * eye2, val)
            return theta - delta, None

        final_theta, _ = lax.scan(body_fn, start_theta, None, length=num_iters)
        return final_theta

    final_points = vmap(refine_candidate)(best_candidates)
    final_betas = ray_trace_fn(final_points)
    final_dists = jnp.linalg.norm(final_betas - source_pos, axis=-1)

    sort_idx = jnp.argsort(final_dists)
    return final_points[sort_idx], final_dists[sort_idx]


def solve_lens_equation_mesh_refine_core(
    source_pos: jnp.ndarray,
    ray_trace_fn: Callable[[jnp.ndarray], jnp.ndarray],
    initial_range: float = 5.0,
    n_x: int = 100,
    n_y: int = 100,
    k_keep: int = 20,
    subgrid_res: int = 20,
    depth: int = 10,
    search_factor: float = 2.0,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Solve lens equation with adaptive mesh refinement around coarse seeds.

    Parameters
    ----------
    source_pos : jnp.ndarray
        Target source position ``(2,)``.
    ray_trace_fn : callable
        Ray-tracing function.
    initial_range : float, optional
        Coarse search half-width.
    n_x, n_y : int, optional
        Coarse-grid resolution.
    k_keep : int, optional
        Number of seeds retained after coarse stage.
    subgrid_res : int, optional
        Refinement grid resolution around each candidate.
    depth : int, optional
        Number of refinement levels.
    search_factor : float, optional
        Factor controlling refinement window shrink rate.

    Returns
    -------
    tuple[jnp.ndarray, jnp.ndarray]
        Refined candidate positions and residual norms, sorted by residual.
    """
    source_pos = jnp.asarray(source_pos, dtype=jnp.float32)
    best_candidates, initial_pixel_width = _find_initial_candidates(
        source_pos=source_pos,
        ray_trace_fn=ray_trace_fn,
        initial_range=initial_range,
        n_x=n_x,
        n_y=n_y,
        k_keep=k_keep,
    )

    current_width = initial_pixel_width * float(search_factor)
    init_val = (best_candidates, current_width)

    def body_fn(carry: Tuple[jnp.ndarray, float], _: None):
        """Refine all candidates by local grid search at current scale."""
        candidates, width = carry

        local_xs = jnp.linspace(-width / 2.0, width / 2.0, subgrid_res + 1)
        local_ys = jnp.linspace(-width / 2.0, width / 2.0, subgrid_res + 1)
        lx, ly = jnp.meshgrid(local_xs, local_ys)
        offsets = jnp.stack([lx.reshape(-1), ly.reshape(-1)], axis=-1)

        def refine_candidate(center: jnp.ndarray) -> jnp.ndarray:
            """Select best local point around one center candidate."""
            subgrid_theta = center + offsets
            betas = ray_trace_fn(subgrid_theta)
            dists = jnp.linalg.norm(betas - source_pos, axis=-1)
            best_idx = jnp.argmin(dists)
            return subgrid_theta[best_idx]

        next_candidates = vmap(refine_candidate)(candidates)
        next_width = width / float(max(subgrid_res, 1)) * float(search_factor)
        return (next_candidates, next_width), None

    n_refine = max(depth - 1, 0)
    final_state, _ = lax.scan(body_fn, init_val, None, length=n_refine)
    final_points, _ = final_state

    final_betas = ray_trace_fn(final_points)
    final_dists = jnp.linalg.norm(final_betas - source_pos, axis=-1)

    sort_idx = jnp.argsort(final_dists)
    return final_points[sort_idx], final_dists[sort_idx]


@jax.jit
def _compute_cluster_mask(
    images: jnp.ndarray,
    dists: jnp.ndarray,
    tolerance: float,
    cluster_tol: float,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Sort and deduplicate candidate images under residual/separation thresholds.

    Parameters
    ----------
    images : jnp.ndarray
        Candidate image coordinates, shape ``(n_candidates, 2)``.
    dists : jnp.ndarray
        Source-plane residual norms, shape ``(n_candidates,)``.
    tolerance : float
        Maximum residual accepted as valid root.
    cluster_tol : float
        Minimum separation required to keep two roots distinct.

    Returns
    -------
    tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]
        Sorted images, sorted residuals, and keep-mask after deduplication.
    """
    if images.shape[0] == 0:
        return images, dists, jnp.zeros(dists.shape, dtype=bool)

    # Residual remains the primary key so the best numerical root represents
    # each cluster. Values within two machine epsilons of zero are numerically
    # equivalent; treating them as ties prevents eager and fused JIT evaluation
    # from selecting different roots because of insignificant rounding noise.
    residual_resolution = 2.0 * jnp.finfo(dists.dtype).eps
    residual_key = jnp.where(
        dists <= residual_resolution,
        jnp.zeros_like(dists),
        dists,
    )
    sort_idx = jnp.lexsort((images[:, 1], images[:, 0], residual_key))
    sorted_images = images[sort_idx]
    sorted_dists = dists[sort_idx]

    def scan_body(mask: jnp.ndarray, idx: jnp.ndarray):
        """Update keep-mask for one sorted candidate."""
        curr_img = sorted_images[idx]
        curr_dist = sorted_dists[idx]
        all_dists = jnp.linalg.norm(sorted_images - curr_img, axis=-1)
        is_duplicate = jnp.any((all_dists < cluster_tol) & mask)
        keep_this = (curr_dist < tolerance) & (~is_duplicate)
        return mask.at[idx].set(keep_this), keep_this

    initial_mask = jnp.zeros(sorted_dists.shape[0], dtype=bool)
    final_mask, _ = lax.scan(scan_body, initial_mask, jnp.arange(sorted_dists.shape[0]))
    return sorted_images, sorted_dists, final_mask


def post_process_images(
    images: jnp.ndarray,
    dists: jnp.ndarray,
    tolerance: float = 1.0e-4,
    cluster_tol: float = 0.05,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Post-process raw candidates by filtering and deduplicating roots.

    Parameters
    ----------
    images : jnp.ndarray
        Candidate image coordinates.
    dists : jnp.ndarray
        Source-plane residual norms for candidates.
    tolerance : float, optional
        Residual acceptance threshold.
    cluster_tol : float, optional
        Spatial deduplication threshold.

    Returns
    -------
    tuple[jnp.ndarray, jnp.ndarray]
        Filtered unique images and corresponding residuals, ordered by
        residual rank with coordinate tie-breakers.
    """
    if images.shape[0] == 0:
        return images, dists

    sorted_images, sorted_dists, final_mask = _compute_cluster_mask(
        images=images,
        dists=dists,
        tolerance=tolerance,
        cluster_tol=cluster_tol,
    )
    unique_images = sorted_images[final_mask]
    unique_dists = sorted_dists[final_mask]

    # Preserve the residual-first order established by ``_compute_cluster_mask``.
    # Its coordinate tie-breakers make equal-residual roots deterministic while
    # keeping prefixes consistent with ``select_unique_images_fixed``.
    return unique_images, unique_dists


def solve_lens_equation_optimization(
    source_pos: jnp.ndarray,
    ray_trace_fn: Callable[[jnp.ndarray], jnp.ndarray],
    initial_range: float = 5.0,
    n_x: int = 100,
    n_y: int = 100,
    k_keep: int = 20,
    num_iters: int = 20,
    tolerance: float = 1.0e-4,
    cluster_tol: float = 0.05,
    jacobian_eps: float = 1.0e-6,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Public optimization-based lens-equation solver.

    Parameters
    ----------
    source_pos : jnp.ndarray
        Target source position ``(2,)``.
    ray_trace_fn : callable
        Ray-tracing function.
    initial_range, tolerance, cluster_tol, jacobian_eps : float, optional
        Numerical control parameters.
    n_x, n_y, k_keep, num_iters : int, optional
        Search-grid and refinement controls.

    Returns
    -------
    tuple[jnp.ndarray, jnp.ndarray]
        Deduplicated image candidates and residuals.
    """
    candidates, dists = solve_lens_equation_optimization_core(
        source_pos=source_pos,
        ray_trace_fn=ray_trace_fn,
        initial_range=initial_range,
        n_x=n_x,
        n_y=n_y,
        k_keep=k_keep,
        num_iters=num_iters,
        jacobian_eps=jacobian_eps,
    )
    return post_process_images(candidates, dists, tolerance=tolerance, cluster_tol=cluster_tol)


def solve_lens_equation_mesh_refine(
    source_pos: jnp.ndarray,
    ray_trace_fn: Callable[[jnp.ndarray], jnp.ndarray],
    initial_range: float = 5.0,
    n_x: int = 100,
    n_y: int = 100,
    k_keep: int = 20,
    subgrid_res: int = 20,
    depth: int = 10,
    search_factor: float = 2.0,
    tolerance: float = 1.0e-4,
    cluster_tol: float = 0.05,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Public AMR-style lens-equation solver.

    Parameters
    ----------
    source_pos : jnp.ndarray
        Target source position ``(2,)``.
    ray_trace_fn : callable
        Ray-tracing function.
    initial_range, search_factor, tolerance, cluster_tol : float, optional
        Numerical control parameters.
    n_x, n_y, k_keep, subgrid_res, depth : int, optional
        Search-grid and AMR controls.

    Returns
    -------
    tuple[jnp.ndarray, jnp.ndarray]
        Deduplicated image candidates and residuals.
    """
    candidates, dists = solve_lens_equation_mesh_refine_core(
        source_pos=source_pos,
        ray_trace_fn=ray_trace_fn,
        initial_range=initial_range,
        n_x=n_x,
        n_y=n_y,
        k_keep=k_keep,
        subgrid_res=subgrid_res,
        depth=depth,
        search_factor=search_factor,
    )
    return post_process_images(candidates, dists, tolerance=tolerance, cluster_tol=cluster_tol)


@partial(jax.jit, static_argnames=('n_select',))
def select_unique_images_and_dists_fixed(
    images: jnp.ndarray,
    dists: jnp.ndarray,
    n_select: int,
    tolerance: float,
    cluster_tol: float,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Select a fixed number of unique valid images from candidates.

    This function filters candidate image positions by checking:
    1. Numerical validity: The distance (residual) must be within `tolerance`.
    2. Uniqueness: The image must not be within `cluster_tol` of already selected images.
    
    It delegates filtering and deduplication to ``_compute_cluster_mask``, then
    stable-partitions valid roots ahead of rejected candidates and pads with a
    dummy zero row, so the output keeps a fixed shape and stays JIT-compilable
    and efficient on GPUs without dynamic array shapes.

    Parameters
    ----------
    images : jnp.ndarray
        Candidate image positions, shape (N, 2).
    dists : jnp.ndarray
        Ray-traced source-plane position residuals for each image candidate, shape (N,).
    n_select : int
        The exact number of images to return. This is a static argument for JIT.
    tolerance : float
        Maximum allowed residual for a candidate to be considered a valid root.
    cluster_tol : float
        Minimum separation between unique images to avoid duplicates.

    Returns
    -------
    selected_images : jnp.ndarray
        Array of selected image positions, shape (n_select, 2). Valid rows
        retain residual-first order with coordinate tie-breakers.
    selected_dists : jnp.ndarray
        Residual norms aligned with ``selected_images``. Invalid rows are zero
        padded.
    selected_mask : jnp.ndarray
        Boolean mask indicating valid entries in selected_images, shape (n_select,).
    count : jnp.ndarray
        Scalar integer indicating the number of valid returned rows.
    """
    sorted_images, sorted_dists, unique_mask = _compute_cluster_mask(
        images=images,
        dists=dists,
        tolerance=tolerance,
        cluster_tol=cluster_tol,
    )

    # ``sorted_images`` is already ordered by residual (with coordinates as
    # exact-tie breakers). Move valid unique roots ahead of rejected candidates
    # without changing that ranking, so truncation retains the best roots.
    candidate_idx = jnp.arange(sorted_images.shape[0])
    priority_idx = jnp.lexsort(
        (candidate_idx, (~unique_mask).astype(jnp.int32))
    )
    ranked_images = sorted_images[priority_idx]
    ranked_dists = sorted_dists[priority_idx]
    ranked_mask = unique_mask[priority_idx]

    # A dummy row supplies fixed-shape zero padding when fewer than
    # ``n_select`` candidates exist.
    n_candidates = ranked_images.shape[0]
    padded_images = jnp.concatenate(
        (
            ranked_images,
            jnp.zeros((1, ranked_images.shape[1]), dtype=ranked_images.dtype),
        ),
        axis=0,
    )
    padded_dists = jnp.concatenate(
        (ranked_dists, jnp.zeros((1,), dtype=ranked_dists.dtype)), axis=0
    )
    padded_mask = jnp.concatenate((ranked_mask, jnp.zeros((1,), dtype=bool)))
    selected_idx = jnp.minimum(jnp.arange(n_select), n_candidates)
    selected_images = padded_images[selected_idx]
    selected_dists = padded_dists[selected_idx]
    selected_mask = padded_mask[selected_idx]

    # Preserve the residual-first order established by
    # ``_compute_cluster_mask`` so this result matches the corresponding
    # prefix from ``post_process_images``. Invalid fixed-shape slots are
    # already ranked last and are normalized to zero padding here.
    selected_images = jnp.where(
        selected_mask[:, None], selected_images, jnp.zeros_like(selected_images)
    )
    selected_dists = jnp.where(
        selected_mask, selected_dists, jnp.zeros_like(selected_dists)
    )
    count = jnp.sum(selected_mask, dtype=jnp.int32)
    return selected_images, selected_dists, selected_mask, count


@partial(jax.jit, static_argnames=('n_select',))
def select_unique_images_fixed(
    images: jnp.ndarray,
    dists: jnp.ndarray,
    n_select: int,
    tolerance: float,
    cluster_tol: float,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Return fixed-shape unique image positions for compatibility callers."""
    selected_images, _, selected_mask, count = select_unique_images_and_dists_fixed(
        images=images,
        dists=dists,
        n_select=n_select,
        tolerance=tolerance,
        cluster_tol=cluster_tol,
    )
    return selected_images, selected_mask, count


def build_permutation_indices(n_points: int) -> jnp.ndarray:
    """
    Build all possible permutation indices for a given number of points.

    This utility is used for assignment-invariant likelihood calculations, where
    the order of observed vs. predicted images is unknown. By precomputing all
    N! permutations, we can efficiently find the minimum chi-square across all
    possible one-to-one assignments on the GPU.

    Parameters
    ----------
    n_points : int
        The number of points (images) to permute.

    Returns
    -------
    permutation_indices : jnp.ndarray
        A 2D array of shape (N!, N) containing all permutation index sequences.

    Examples
    --------
    >>> build_permutation_indices(3)
    Array([[0, 1, 2],
           [0, 2, 1],
           [1, 0, 2],
           [1, 2, 0],
           [2, 0, 1],
           [2, 1, 0]], dtype=int32)

    Notes
    -----
    The number of permutations grows factorially (N!). To avoid memory issues
    and combinatorial explosion, this is restricted to N <= 8. For larger N,
    the Hungarian algorithm (`min_assignment_chi2_hungarian`) should be used.
    """
    n_points = int(n_points)
    if n_points < 1:
        raise ValueError("n_points must be >= 1")
    if n_points > 8:
        raise ValueError("n_points > 8 is not supported due combinatorial explosion (N! growth)")

    perm = np.array(list(itertools.permutations(range(n_points))), dtype=np.int32)
    return jnp.asarray(perm)


@jax.jit
def min_assignment_chi2(
    observed_positions: jnp.ndarray,
    predicted_positions: jnp.ndarray,
    sigma_pos: jnp.ndarray,
    permutation_indices: jnp.ndarray,
) -> jnp.ndarray:
    """
    Compute the minimum chi-square by testing all possible image assignments.

    In strong lensing, the solver might return images in a different order than 
    the observations. This function solves the "assignment problem" by brute-force
    evaluating the chi-square for every possible permutation of the predicted 
    images and returning the minimum.

    This approach is highly efficient on GPUs for a small number of images (N <= 4)
    because it uses vectorized matrix operations instead of iterative algorithms.

    Parameters
    ----------
    observed_positions : jnp.ndarray
        The observed image positions, shape (N, 2).
    predicted_positions : jnp.ndarray
        The predicted image positions from the lens model, shape (N, 2).
    sigma_pos : jnp.ndarray
        The 1D array of positional uncertainties for each observed image, shape (N,).
    permutation_indices : jnp.ndarray
        Precomputed permutation indices of shape (N!, N), usually from 
        `build_permutation_indices`.

    Returns
    -------
    min_chi2 : jnp.ndarray
        The minimum chi-square value across all possible assignments.
    """
    # Small epsilon to avoid division by zero
    sigma2 = jnp.square(sigma_pos) + 1.0e-12

    # Compute the cost matrix: C_ij = (obs_i - pred_j)^2 / sigma_i^2
    # residual shape: (N_obs, N_pred, 2)
    residual = observed_positions[:, None, :] - predicted_positions[None, :, :]
    # sqdist shape: (N_obs, N_pred)
    sqdist = jnp.sum(jnp.square(residual), axis=-1)
    # cost shape: (N_obs, N_pred)
    cost = sqdist / sigma2[:, None]

    # Evaluate cost for all N! permutations via advanced indexing
    obs_idx = jnp.arange(observed_positions.shape[0])[None, :]
    perm_cost = jnp.sum(cost[obs_idx, permutation_indices], axis=1)
    return jnp.min(perm_cost)


@jax.jit
def min_assignment_chi2_subset(
    observed_positions: jnp.ndarray,
    predicted_positions: jnp.ndarray,
    sigma_pos: jnp.ndarray,
    predicted_valid: jnp.ndarray,
) -> jnp.ndarray:
    """Match observations to the best one-to-one subset of predicted roots.

    Extra predicted roots are skipped without penalty. The dynamic program has
    one state per subset of observations already assigned, which is practical
    for the small point-image multiplicities handled by this GPU path.
    """
    n_observed = observed_positions.shape[0]
    n_states = 1 << n_observed
    sigma2 = jnp.square(sigma_pos) + 1.0e-12

    residual = observed_positions[:, None, :] - predicted_positions[None, :, :]
    sqdist = jnp.sum(jnp.square(residual), axis=-1)
    cost_matrix = sqdist / sigma2[:, None]
    cost_matrix = jnp.where(predicted_valid[None, :], cost_matrix, jnp.inf)

    state_masks = jnp.arange(n_states, dtype=jnp.int32)
    initial_costs = jnp.full((n_states,), jnp.inf, dtype=cost_matrix.dtype)
    initial_costs = initial_costs.at[0].set(0.0)

    def process_candidate(candidate_idx: int, costs: jnp.ndarray) -> jnp.ndarray:
        def assign_observation(observed_idx: int, next_costs: jnp.ndarray) -> jnp.ndarray:
            bit = jnp.left_shift(jnp.int32(1), jnp.int32(observed_idx))
            destinations = jnp.bitwise_or(state_masks, bit)
            can_assign = jnp.equal(jnp.bitwise_and(state_masks, bit), 0)
            proposals = jnp.where(
                can_assign,
                costs + cost_matrix[observed_idx, candidate_idx],
                jnp.inf,
            )
            return next_costs.at[destinations].min(proposals)

        return lax.fori_loop(0, n_observed, assign_observation, costs)

    final_costs = lax.fori_loop(
        0, predicted_positions.shape[0], process_candidate, initial_costs
    )
    return final_costs[-1]


def _hungarian_assignment_callback(cost_matrix):
    """
    NumPy/SciPy callback for Hungarian assignment.

    Parameters
    ----------
    cost_matrix : np.ndarray
        Assignment cost matrix with shape ``(n_obs, n_pred)``.

    Returns
    -------
    np.ndarray
        Best matching predicted-index permutation for rows ``0..n_obs-1``.
    """
    row_ind, col_ind = scipy.optimize.linear_sum_assignment(cost_matrix)
    # Sort by row_ind to ensure we get the permutation for rows 0, 1, 2...
    sort_idx = np.argsort(row_ind)
    return col_ind[sort_idx].astype(np.int32)


@jax.jit
def min_assignment_chi2_hungarian(
    observed_positions: jnp.ndarray,
    predicted_positions: jnp.ndarray,
    sigma_pos: jnp.ndarray,
    predicted_valid: Optional[jnp.ndarray] = None,
) -> jnp.ndarray:
    """
    Compute minimum assignment chi-square using Hungarian algorithm.

    Parameters
    ----------
    observed_positions : jnp.ndarray
        Observed image positions, shape ``(n_obs, 2)``.
    predicted_positions : jnp.ndarray
        Predicted image positions, shape ``(n_pred, 2)``.
    sigma_pos : jnp.ndarray
        Positional 1-sigma uncertainties, shape ``(n_obs,)``.
    predicted_valid : jnp.ndarray, optional
        Mask for valid rows in ``predicted_positions``. Invalid rows cannot be
        selected by the assignment.

    Returns
    -------
    jnp.ndarray
        Minimum chi-square value over one-to-one assignments.
    """
    sigma2 = jnp.square(sigma_pos) + 1.0e-12

    # Cost matrix C_ij = chi2 contribution of assigning obs i to pred j
    residual = observed_positions[:, None, :] - predicted_positions[None, :, :]
    sqdist = jnp.sum(jnp.square(residual), axis=-1)
    cost_matrix = sqdist / sigma2[:, None]
    if predicted_valid is not None:
        cost_matrix = jnp.where(predicted_valid[None, :], cost_matrix, jnp.inf)

    n_obs = observed_positions.shape[0]
    
    # Use pure_callback to call scipy's Hungarian algorithm
    # We need the optimal permutation indices to sum the costs
    # Use stop_gradient on cost_matrix to prevent JAX from trying to differentiate through pure_callback
    col_ind = jax.pure_callback(
        _hungarian_assignment_callback,
        jnp.zeros(n_obs, dtype=jnp.int32),  # return shape info
        lax.stop_gradient(cost_matrix),
    )
    
    # Stop gradient on indices because the assignment is discrete
    col_ind = lax.stop_gradient(col_ind)
    
    # Gather the costs using the optimal assignment
    # total_cost = sum(cost_matrix[i, col_ind[i]])
    row_ind = jnp.arange(n_obs)
    min_chi2 = jnp.sum(cost_matrix[row_ind, col_ind])
    
    return min_chi2


__all__ = [
    'solve_lens_equation_optimization_core',
    'solve_lens_equation_mesh_refine_core',
    'solve_lens_equation_optimization',
    'solve_lens_equation_mesh_refine',
    'post_process_images',
    'select_unique_images_fixed',
    'build_permutation_indices',
    'min_assignment_chi2',
    'min_assignment_chi2_subset',
    'min_assignment_chi2_hungarian',
]
