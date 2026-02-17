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
from typing import Callable, Tuple

import jax
import numpy as np
import scipy.optimize
from jax import jacfwd, lax, vmap
import jax.numpy as jnp


def _ray_trace_from_fn(theta: jnp.ndarray, ray_trace_fn: Callable[[jnp.ndarray], jnp.ndarray]) -> jnp.ndarray:
    """
    Evaluate user-provided ray-tracing callable.

    Parameters
    ----------
    theta : jnp.ndarray
        Image-plane coordinates with shape ``(..., 2)``.
    ray_trace_fn : callable
        Function mapping image-plane coordinates to source-plane coordinates.

    Returns
    -------
    jnp.ndarray
        Source-plane coordinates with shape ``(..., 2)``.
    """
    return ray_trace_fn(theta)


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
    xs = jnp.linspace(-initial_range, initial_range, int(n_x) + 1)
    ys = jnp.linspace(-initial_range, initial_range, int(n_y) + 1)
    grid_x, grid_y = jnp.meshgrid(xs, ys)
    initial_theta = jnp.stack([grid_x.reshape(-1), grid_y.reshape(-1)], axis=-1)

    beta_0 = _ray_trace_from_fn(initial_theta, ray_trace_fn)
    dist_0 = jnp.linalg.norm(beta_0 - source_pos, axis=-1)

    dist_grid = dist_0.reshape(int(n_y) + 1, int(n_x) + 1)
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
    k_keep = int(min(max(k_keep, 1), n_total))
    _, indices = lax.top_k(scores, k_keep)

    pixel_width = (2.0 * float(initial_range)) / float(max(int(n_x), 1))
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
                return _ray_trace_from_fn(t, ray_trace_fn) - source_pos

            val = f(theta)
            jac = jacfwd(f)(theta)
            delta = jnp.linalg.solve(jac + jacobian_eps * eye2, val)
            return theta - delta, None

        final_theta, _ = lax.scan(body_fn, start_theta, None, length=int(num_iters))
        return final_theta

    final_points = vmap(refine_candidate)(best_candidates)
    final_betas = _ray_trace_from_fn(final_points, ray_trace_fn)
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

        local_xs = jnp.linspace(-width / 2.0, width / 2.0, int(subgrid_res) + 1)
        local_ys = jnp.linspace(-width / 2.0, width / 2.0, int(subgrid_res) + 1)
        lx, ly = jnp.meshgrid(local_xs, local_ys)
        offsets = jnp.stack([lx.reshape(-1), ly.reshape(-1)], axis=-1)

        def refine_candidate(center: jnp.ndarray) -> jnp.ndarray:
            """Select best local point around one center candidate."""
            subgrid_theta = center + offsets
            betas = _ray_trace_from_fn(subgrid_theta, ray_trace_fn)
            dists = jnp.linalg.norm(betas - source_pos, axis=-1)
            best_idx = jnp.argmin(dists)
            return subgrid_theta[best_idx]

        next_candidates = vmap(refine_candidate)(candidates)
        next_width = width / float(max(int(subgrid_res), 1)) * float(search_factor)
        return (next_candidates, next_width), None

    n_refine = max(int(depth) - 1, 0)
    final_state, _ = lax.scan(body_fn, init_val, None, length=n_refine)
    final_points, _ = final_state

    final_betas = _ray_trace_from_fn(final_points, ray_trace_fn)
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
    sort_idx = jnp.argsort(dists)
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
        Filtered unique images and corresponding residuals.
    """
    if images.shape[0] == 0:
        return images, dists

    sorted_images, sorted_dists, final_mask = _compute_cluster_mask(
        images=images,
        dists=dists,
        tolerance=tolerance,
        cluster_tol=cluster_tol,
    )
    return sorted_images[final_mask], sorted_dists[final_mask]


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
def select_unique_images_fixed(
    images: jnp.ndarray,
    dists: jnp.ndarray,
    n_select: int,
    tolerance: float,
    cluster_tol: float,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Select a fixed number of unique valid images from candidates.

    This function filters candidate image positions by checking:
    1. Numerical validity: The distance (residual) must be within `tolerance`.
    2. Uniqueness: The image must not be within `cluster_tol` of already selected images.
    
    It uses a fixed output shape and JAX control flow (`lax.scan`, `lax.cond`) to ensure
    the function is JIT-compilable and efficient on GPUs, avoiding dynamic array shapes.

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
        Array of selected image positions, shape (n_select, 2).
    selected_mask : jnp.ndarray
        Boolean mask indicating valid entries in selected_images, shape (n_select,).
    count : jnp.ndarray
        Scalar integer indicating the total number of valid images found.
    """
    # Sort by residual distance to prioritize better solutions
    sort_idx = jnp.argsort(dists)
    sorted_images = images[sort_idx]
    sorted_dists = dists[sort_idx]

    n_select = int(n_select)
    init_images = jnp.zeros((n_select, 2), dtype=sorted_images.dtype)
    init_mask = jnp.zeros((n_select,), dtype=bool)
    init_count = jnp.array(0, dtype=jnp.int32)

    def body_fn(carry, idx):
        """Scan one candidate and update fixed-size selected set."""
        selected, selected_mask, count = carry
        curr_img = sorted_images[idx]
        curr_dist = sorted_dists[idx]

        # Check for duplicates against all previously selected images
        sep = jnp.linalg.norm(selected - curr_img, axis=-1)
        duplicated = jnp.any(jnp.logical_and(selected_mask, sep < cluster_tol))

        # Filtering conditions
        valid_dist = curr_dist < tolerance
        has_slot = count < n_select
        can_add = jnp.logical_and(jnp.logical_and(valid_dist, ~duplicated), has_slot)

        # Update state using JAX conditional updates to keep shapes static
        selected = lax.cond(
            can_add,
            lambda arr: arr.at[count].set(curr_img),
            lambda arr: arr,
            selected,
        )
        selected_mask = lax.cond(
            can_add,
            lambda arr: arr.at[count].set(True),
            lambda arr: arr,
            selected_mask,
        )
        count = count + can_add.astype(jnp.int32)

        return (selected, selected_mask, count), None

    # Scan through all sorted candidates
    final_state, _ = lax.scan(
        body_fn,
        (init_images, init_mask, init_count),
        jnp.arange(sorted_images.shape[0]),
    )
    return final_state


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

    # Use the precomputed permutations to sum up costs for every assignment.
    # We use JAX advanced indexing and broadcasting to compute all permutations at once.
    # obs_idx: [1, N] array containing [[0, 1, ..., N-1]]
    obs_idx = jnp.arange(observed_positions.shape[0])[None, :]
    
    # cost[obs_idx, permutation_indices] uses advanced indexing:
    # - obs_idx has shape (1, N) and provides the observed indices [0..N-1].
    # - permutation_indices has shape (N!, N); each row is one assignment of predicted indices.
    # Broadcasting produces an array of shape (N!, N) where:
    #   selected[i, j] = cost[j, permutation_indices[i, j]]
    # i.e. row i lists the per-image costs for the i-th one-to-one assignment.
    #
    # Example with N=3 (3 observed, 3 predicted images):
    #   obs_idx = [[0, 1, 2]]  # shape (1, 3)
    #   permutation_indices = [[0, 1, 2],  # shape (6, 3), all 3! = 6 permutations
    #                         [0, 2, 1],
    #                         [1, 0, 2],
    #                         [1, 2, 0],
    #                         [2, 0, 1],
    #                         [2, 1, 0]]
    #   cost = [[0.1, 0.5, 0.9],  # shape (3, 3): cost[i,j] = cost matching observed i with predicted j
    #           [0.2, 0.6, 1.0],
    #           [0.3, 0.7, 1.1]]
    #   cost[obs_idx, permutation_indices] = [[0.1, 0.6, 1.1],  # shape (6, 3): each row is one assignment's costs
    #                                         [0.1, 1.0, 0.7],
    #                                         [0.5, 0.2, 1.1],
    #                                         [0.5, 1.0, 0.3],
    #                                         [0.9, 0.2, 0.7],
    #                                         [0.9, 0.6, 0.3]]
    #   perm_cost = [1.8, 1.8, 1.8, 1.8, 1.8, 1.8]  # shape (6,): sum each row to get total cost per assignment
    
    # Sum along axis 1 to get total chi-square for each of the N! permutations.
    # perm_cost shape: (N_permutations,)
    perm_cost = jnp.sum(cost[obs_idx, permutation_indices], axis=1)
    
    # Return the global minimum cost (best assignment)
    return jnp.min(perm_cost)


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
) -> jnp.ndarray:
    """
    Compute minimum assignment chi-square using Hungarian algorithm.

    Parameters
    ----------
    observed_positions : jnp.ndarray
        Observed image positions, shape ``(n_obs, 2)``.
    predicted_positions : jnp.ndarray
        Predicted image positions, shape ``(n_obs, 2)``.
    sigma_pos : jnp.ndarray
        Positional 1-sigma uncertainties, shape ``(n_obs,)``.

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
    'min_assignment_chi2_hungarian',
]
