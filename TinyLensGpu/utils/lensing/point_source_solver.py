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
from jax import jacfwd, lax, vmap
import jax.numpy as jnp


def _ray_trace_from_fn(theta: jnp.ndarray, ray_trace_fn: Callable[[jnp.ndarray], jnp.ndarray]) -> jnp.ndarray:
    """Evaluate lens equation mapping for stacked image-plane coordinates."""
    return ray_trace_fn(theta)


def _find_initial_candidates(
    source_pos: jnp.ndarray,
    ray_trace_fn: Callable[[jnp.ndarray], jnp.ndarray],
    initial_range: float,
    n_x: int,
    n_y: int,
    k_keep: int,
) -> Tuple[jnp.ndarray, float]:
    """Find coarse image-plane candidates with local-minimum preference."""
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
    """Solve lens equation with coarse search + Newton refinement."""
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
        def body_fn(theta: jnp.ndarray, _: None) -> Tuple[jnp.ndarray, None]:
            def f(t: jnp.ndarray) -> jnp.ndarray:
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
    """Solve lens equation with static-budget adaptive mesh refinement."""
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
        candidates, width = carry

        local_xs = jnp.linspace(-width / 2.0, width / 2.0, int(subgrid_res) + 1)
        local_ys = jnp.linspace(-width / 2.0, width / 2.0, int(subgrid_res) + 1)
        lx, ly = jnp.meshgrid(local_xs, local_ys)
        offsets = jnp.stack([lx.reshape(-1), ly.reshape(-1)], axis=-1)

        def refine_candidate(center: jnp.ndarray) -> jnp.ndarray:
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
    """Compute boolean mask for unique valid image solutions."""
    sort_idx = jnp.argsort(dists)
    sorted_images = images[sort_idx]
    sorted_dists = dists[sort_idx]

    def scan_body(mask: jnp.ndarray, idx: jnp.ndarray):
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
    """Filter and cluster lens-equation candidates (dynamic output length)."""
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
    """High-level optimization solver returning clustered image solutions."""
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
    """High-level AMR solver returning clustered image solutions."""
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
    """Select up to n_select unique valid images with fixed output shape."""
    sort_idx = jnp.argsort(dists)
    sorted_images = images[sort_idx]
    sorted_dists = dists[sort_idx]

    n_select = int(n_select)
    init_images = jnp.zeros((n_select, 2), dtype=sorted_images.dtype)
    init_mask = jnp.zeros((n_select,), dtype=bool)
    init_count = jnp.array(0, dtype=jnp.int32)

    def body_fn(carry, idx):
        selected, selected_mask, count = carry
        curr_img = sorted_images[idx]
        curr_dist = sorted_dists[idx]

        sep = jnp.linalg.norm(selected - curr_img, axis=-1)
        duplicated = jnp.any(jnp.logical_and(selected_mask, sep < cluster_tol))

        valid_dist = curr_dist < tolerance
        has_slot = count < n_select
        can_add = jnp.logical_and(jnp.logical_and(valid_dist, ~duplicated), has_slot)

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

    final_state, _ = lax.scan(
        body_fn,
        (init_images, init_mask, init_count),
        jnp.arange(sorted_images.shape[0]),
    )
    return final_state


def build_permutation_indices(n_points: int) -> jnp.ndarray:
    """Build all permutation indices for assignment-invariant matching."""
    n_points = int(n_points)
    if n_points < 1:
        raise ValueError("n_points must be >= 1")
    if n_points > 8:
        raise ValueError("n_points > 8 is not supported due combinatorial explosion")

    perm = np.array(list(itertools.permutations(range(n_points))), dtype=np.int32)
    return jnp.asarray(perm)


@jax.jit
def min_assignment_chi2(
    observed_positions: jnp.ndarray,
    predicted_positions: jnp.ndarray,
    sigma_pos: jnp.ndarray,
    permutation_indices: jnp.ndarray,
) -> jnp.ndarray:
    """Compute minimum chi-square under all one-to-one assignments."""
    sigma2 = jnp.square(sigma_pos) + 1.0e-12

    residual = observed_positions[:, None, :] - predicted_positions[None, :, :]
    sqdist = jnp.sum(jnp.square(residual), axis=-1)
    cost = sqdist / sigma2[:, None]

    obs_idx = jnp.arange(observed_positions.shape[0])[None, :]
    perm_cost = jnp.sum(cost[obs_idx, permutation_indices], axis=1)
    return jnp.min(perm_cost)


__all__ = [
    'solve_lens_equation_optimization_core',
    'solve_lens_equation_mesh_refine_core',
    'solve_lens_equation_optimization',
    'solve_lens_equation_mesh_refine',
    'post_process_images',
    'select_unique_images_fixed',
    'build_permutation_indices',
    'min_assignment_chi2',
]

