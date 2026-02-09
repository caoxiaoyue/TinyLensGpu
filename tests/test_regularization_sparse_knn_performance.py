"""Performance benchmark for sparse KNN regularization under duplicated points.

This benchmark compares:
1) current implementation (self-diagonal masked before top-k), and
2) legacy implementation (top-k with k+1 then drop first column).

The focus is a high-duplicate-points scenario where self-selection bugs were
observed in the legacy approach.
"""

from __future__ import annotations

import time
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import random

from TinyLensGpu.utils.lensing.regularization import regularization_sparse_knn_from


def _kernel_weight_jax(distance: jnp.ndarray, scale: float, reg_type: str) -> jnp.ndarray:
    scale = jnp.maximum(scale, 1e-6)

    if reg_type == "exp":
        return jnp.exp(-distance / scale)
    if reg_type == "gauss":
        return jnp.exp(-(distance**2) / (2.0 * scale * scale))
    if reg_type == "matern32":
        sqrt3 = jnp.sqrt(3.0)
        z = sqrt3 * distance / scale
        return (1.0 + z) * jnp.exp(-z)
    if reg_type == "matern52":
        sqrt5 = jnp.sqrt(5.0)
        z = sqrt5 * distance / scale
        return (1.0 + z + (z * z) / 3.0) * jnp.exp(-z)

    raise ValueError(f"Unknown reg_type: '{reg_type}'")


@partial(jax.jit, static_argnames=["reg_type", "k_neighbors"])
def _legacy_regularization_sparse_knn_from(
    scale: float,
    coefficient: float,
    points: jnp.ndarray,
    reg_type: str = "exp",
    k_neighbors: int = 16,
):
    """Legacy version for benchmark only (pre-fix behavior)."""
    n_source = points.shape[0]

    if n_source == 0:
        return (
            jnp.zeros((0,), dtype=jnp.int32),
            jnp.zeros((0,), dtype=jnp.int32),
            jnp.zeros((0,), dtype=jnp.float32),
            0,
        )

    if n_source == 1:
        diag = jnp.array([max(float(coefficient), 1e-6)], dtype=jnp.float32)
        return (
            jnp.array([0], dtype=jnp.int32),
            jnp.array([0], dtype=jnp.int32),
            diag,
            1,
        )

    diff = points[:, None, :] - points[None, :, :]
    dist_sq = jnp.sum(diff**2, axis=-1)
    dist = jnp.sqrt(dist_sq + 1e-12)

    k = max(1, min(int(k_neighbors), int(n_source) - 1))
    search_k = k + 1

    neg_dist = -dist
    top_vals, top_idx = jax.lax.top_k(neg_dist, search_k)

    neighbors_dist = -top_vals[:, 1:]
    neighbors_idx = top_idx[:, 1:]

    weights = _kernel_weight_jax(neighbors_dist, scale, reg_type)
    weights = weights * coefficient

    row_indices = jnp.repeat(jnp.arange(n_source), k)
    col_indices = neighbors_idx.flatten()
    edge_weights = weights.flatten()

    all_rows_off = jnp.concatenate([row_indices, col_indices])
    all_cols_off = jnp.concatenate([col_indices, row_indices])
    all_vals_off = jnp.concatenate([-edge_weights, -edge_weights])

    diag_sum = jax.ops.segment_sum(-all_vals_off, all_rows_off, num_segments=n_source)
    ridge = jnp.maximum(1e-8, 1e-6 * jnp.maximum(1.0, coefficient))
    diag_vals = diag_sum + ridge

    diag_rows = jnp.arange(n_source)
    diag_cols = jnp.arange(n_source)

    final_rows = jnp.concatenate([all_rows_off, diag_rows])
    final_cols = jnp.concatenate([all_cols_off, diag_cols])
    final_vals = jnp.concatenate([all_vals_off, diag_vals])

    return (
        final_rows.astype(jnp.int32),
        final_cols.astype(jnp.int32),
        final_vals.astype(jnp.float32),
        n_source,
    )


def _make_points_with_duplicates(n_points: int, duplicate_ratio: float, seed: int = 0) -> jnp.ndarray:
    duplicate_ratio = float(np.clip(duplicate_ratio, 0.0, 0.95))
    n_unique = max(1, int(round(n_points * (1.0 - duplicate_ratio))))
    n_duplicate = n_points - n_unique

    key = random.PRNGKey(seed)
    key_u, key_i, key_p = random.split(key, 3)
    unique_points = random.normal(key_u, (n_unique, 2), dtype=jnp.float32)

    if n_duplicate > 0:
        dup_indices = random.randint(key_i, (n_duplicate,), minval=0, maxval=n_unique)
        duplicate_points = unique_points[dup_indices]
        points = jnp.concatenate([unique_points, duplicate_points], axis=0)
    else:
        points = unique_points

    perm = random.permutation(key_p, jnp.arange(n_points))
    return points[perm]


def _time_sparse_builder(fn, *, points, scale, coefficient, reg_type, k_neighbors, n_runs=5):
    # warmup/compile
    warm = fn(scale, coefficient, points, reg_type=reg_type, k_neighbors=k_neighbors)
    warm[2].block_until_ready()

    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        out = fn(scale, coefficient, points, reg_type=reg_type, k_neighbors=k_neighbors)
        out[2].block_until_ready()
        times.append(time.perf_counter() - t0)

    arr = np.asarray(times, dtype=np.float64)
    return {
        "mean_s": float(arr.mean()),
        "std_s": float(arr.std(ddof=1) if n_runs > 1 else 0.0),
        "min_s": float(arr.min()),
        "max_s": float(arr.max()),
        "n_runs": int(n_runs),
    }


@pytest.mark.performance
@pytest.mark.slow
def test_sparse_knn_fix_vs_legacy_runtime_high_duplicate_ratio():
    """Benchmark runtime change of fix vs legacy in high-duplicate-points case."""
    n_points = 1800
    duplicate_ratio = 0.70
    scale = 0.1
    coefficient = 1.0
    reg_type = "exp"
    k_neighbors = 16

    points = _make_points_with_duplicates(n_points=n_points, duplicate_ratio=duplicate_ratio, seed=7)

    fixed_stats = _time_sparse_builder(
        regularization_sparse_knn_from,
        points=points,
        scale=scale,
        coefficient=coefficient,
        reg_type=reg_type,
        k_neighbors=k_neighbors,
        n_runs=4,
    )
    legacy_stats = _time_sparse_builder(
        _legacy_regularization_sparse_knn_from,
        points=points,
        scale=scale,
        coefficient=coefficient,
        reg_type=reg_type,
        k_neighbors=k_neighbors,
        n_runs=4,
    )

    ratio = fixed_stats["mean_s"] / max(legacy_stats["mean_s"], 1e-12)

    # Print machine-readable result for quick comparison in CI/local logs.
    print(
        {
            "n_points": n_points,
            "duplicate_ratio": duplicate_ratio,
            "k_neighbors": k_neighbors,
            "fixed": fixed_stats,
            "legacy": legacy_stats,
            "fixed_over_legacy": ratio,
        }
    )

    # Guardrail: fix should not introduce severe slowdown.
    # Keep threshold loose to avoid flaky failures across different devices.
    assert ratio <= 1.60, (
        "Sparse-KNN fix is unexpectedly slower under high duplicates: "
        f"fixed={fixed_stats['mean_s']:.4f}s legacy={legacy_stats['mean_s']:.4f}s ratio={ratio:.3f}"
    )

