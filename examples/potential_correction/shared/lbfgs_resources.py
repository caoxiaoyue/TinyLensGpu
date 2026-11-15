"""Synchronized first-fit resource measurements for both L-BFGS-B backends."""

import os
from time import perf_counter

import jax
import numpy as np
import scipy


def measure_lbfgs_fit(problem, strengths, **fit_options):
    """Fit a fresh problem and measure it in a backend-isolated process.

    Stage timings include any compilation incurred on the first call. The
    device allocator peak is cumulative from device initialization through
    the synchronized fit and includes the preparation performed by the caller.
    """
    started = perf_counter()
    result = problem.fit(strengths, **fit_options)
    jax.block_until_ready(
        (
            result.source_values,
            result.potential_values,
            result.nonlinear_model_image,
            result.log_evidence,
        )
    )
    fit_seconds = perf_counter() - started
    device = next(iter(result.source_values.devices()))
    memory_stats = device.memory_stats() or {}
    return result, {
        "device": device.device_kind,
        "platform": device.platform,
        "precision": result.dtype,
        "jax": jax.__version__,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "xla_preallocate": os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE"),
        "cold_fit_seconds_including_compilation": fit_seconds,
        "fit_stage_seconds": dict(result.configuration["timings"]),
        "peak_device_bytes_in_use": memory_stats.get("peak_bytes_in_use"),
        "peak_memory_method": (
            "JAX device allocator peak_bytes_in_use from initialization through the synchronized fit"
        ),
    }
