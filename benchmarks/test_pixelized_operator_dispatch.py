"""Guard direct operator calls against losing their fused JAX kernels."""

import json
from statistics import median
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from TinyLensGpu.ForwardSimulation import SimulatorConfig
from TinyLensGpu.ForwardSimulation.LensImage.pixelized_operator import (
    PixelizedLensOperator,
)
from TinyLensGpu.PhysicalModel import PhysicalModel, PixelizedSourceModel, SersicEllipse
from TinyLensGpu.utils.inversion.regularization import FiniteDifferenceRegularization


def _warm_median(action, argument, repeats=20):
    for _ in range(3):
        jax.block_until_ready(action(argument))
    samples = []
    for _ in range(repeats):
        start = perf_counter()
        jax.block_until_ready(action(argument))
        samples.append(perf_counter() - start)
    return median(samples)


@pytest.mark.performance
@pytest.mark.parametrize("joint", [False, True], ids=["source", "joint"])
def test_direct_curvature_dispatch_stays_close_to_outer_jit(joint, record_property):
    """Direct public matvec calls must retain GPU fusion without a caller JIT."""
    if jax.devices()[0].platform != "gpu":
        pytest.skip("CUDA dispatch performance requires a GPU")
    with jax.experimental.enable_x64():
        size, n, nsub = 40, 20, 2
        y, x = np.indices((size, size))
        mask = (x < 5) | (x > 34) | (y < 5) | (y > 34)
        psf = np.array([[0.02, 0.08, 0.03], [0.1, 0.57, 0.06], [0.04, 0.07, 0.03]])
        source = PixelizedSourceModel(n, regularization_type="first-order")
        source.log_lambda_reg.to_static(0.0)
        lens = SersicEllipse(
            R_sersic=0.3,
            n_sersic=1.0,
            Ie=1.0,
            e1=0.05,
            e2=-0.02,
            center_x=0.02,
            center_y=0.0,
        )
        lens.Ie.to_static(1.0)
        physical = PhysicalModel(
            lens_mass=[], source_light=[source], lens_light=[lens] if joint else []
        )
        config = SimulatorConfig(0.05, size, psf, nsub=nsub, mask=mask)
        simulator = PixelizedLensOperator(physical, config)
        bbox = tuple(np.float64(value) for value in (-1.2, 1.2, -1.2, 1.2))
        noise = jnp.full(simulator.n_active, 0.08, dtype=jnp.float64)
        regularizer = FiniteDifferenceRegularization(n, "first-order")
        reg_data = regularizer.make_reg_data(*bbox)
        mapping = simulator.precompute_operator_data(*bbox)
        if joint:
            operator, _, _ = simulator.build_joint_system(
                noise,
                jnp.ones_like(noise),
                *bbox,
                0.08,
                reg_data,
                1e-3,
                op_data=mapping,
            )
        else:
            operator = simulator.build_source_curvature_operator(
                noise,
                *bbox,
                0.08,
                reg_data,
                op_data=mapping,
            )
        vector = jnp.sin(jnp.arange(operator.size, dtype=jnp.float64) * 0.13)
        fused = jax.jit(operator.matvec)
        np.testing.assert_allclose(
            operator.matvec(vector), fused(vector), rtol=1e-12, atol=1e-12
        )
        direct_seconds = _warm_median(operator.matvec, vector)
        fused_seconds = _warm_median(fused, vector)
    ratio = direct_seconds / fused_seconds
    record_property(
        "dispatch_timings",
        json.dumps(
            dict(
                direct_seconds=direct_seconds,
                outer_jit_seconds=fused_seconds,
                ratio=ratio,
            )
        ),
    )
    # A generous relative guard avoids hardware-specific absolute deadlines.
    # The unfused regression is roughly 30x slower on the reference GPU.
    assert ratio < 6.0, (
        f"direct matvec dispatch lost fusion: {direct_seconds * 1e3:.3f} ms vs "
        f"{fused_seconds * 1e3:.3f} ms with an outer JIT ({ratio:.1f}x)"
    )
