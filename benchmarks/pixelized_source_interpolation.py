"""Measure the matrix-free source solve with a fixed 100-by-100 grid on CUDA."""

import argparse
import json
import platform
from statistics import median
from time import perf_counter

import jax
import jax.numpy as jnp
import jaxlib
import numpy as np

from TinyLensGpu.ForwardSimulation import SimulatorConfig
from TinyLensGpu.ForwardSimulation.LensImage.pixelized_operator import (
    PixelizedLensOperator,
)
from TinyLensGpu.PhysicalModel import PhysicalModel, PixelizedSourceModel, SIE
from TinyLensGpu.utils.cg_solver import pcg_solve
from TinyLensGpu.utils.inversion.regularization import FiniteDifferenceRegularization


def _timed(action, *args):
    start = perf_counter()
    result = jax.block_until_ready(action(*args))
    return result, perf_counter() - start


def _measure(function, args, repeats, device):
    # Compilation and first execution can each incur a separate one-time cost.
    jax.block_until_ready(args)
    start = perf_counter()
    lowered = function.lower(*args)
    lower_seconds = perf_counter() - start
    start = perf_counter()
    compiled = lowered.compile()
    compile_seconds = perf_counter() - start
    _, first = _timed(compiled, *args)
    samples = []
    for _ in range(repeats):
        result, seconds = _timed(compiled, *args)
        samples.append(seconds)
    return result, dict(
        lower_seconds=lower_seconds,
        compile_seconds=compile_seconds,
        first_execution_seconds=first,
        warm_seconds=samples,
        warm_median_seconds=median(samples),
        allocator_peak_to_date_bytes=device.memory_stats()["peak_bytes_in_use"],
    )


def run(interpolation, repeats):
    device = jax.devices("cuda")[0]
    npix, n, nsub, dpix = 100, 100, 4, 0.05
    bbox = (-1.2, 1.2, -1.2, 1.2)
    block_size, strength, rtol, max_iter = 10, 0.08, 1e-5, 400
    axis = (np.arange(npix) - (npix - 1) / 2) * dpix
    x, y = np.meshgrid(axis, axis)
    radius = np.hypot(x, y)
    mask = (radius > 1.8) | (radius < 0.2)
    psf = np.array(
        [[0.01, 0.10, 0.02], [0.09, 0.56, 0.04], [0.01, 0.13, 0.04]],
        dtype=np.float32,
    )
    mass = dict(theta_E=0.8, e1=0.08, e2=-0.04, center_x=0.02, center_y=-0.03)
    lens = SIE(**mass)
    for name in mass:
        getattr(lens, name).to_static()
    source = PixelizedSourceModel(
        n, regularization_type="first-order", interpolation=interpolation,
    )
    physical = PhysicalModel(lens_mass=[lens], source_light=[source], lens_light=[])
    config = SimulatorConfig(dpix, npix, psf, nsub=nsub, mask=mask)
    simulator = PixelizedLensOperator(physical, config)
    regularizer = FiniteDifferenceRegularization(n, source.regularization_type)
    noise = jnp.asarray((0.08 + 0.02 * np.abs(x))[~mask], dtype=jnp.float32)
    image = np.exp(-((radius - 0.87) / 0.22) ** 2) * (
        1 + 0.2 * np.sin(3 * np.arctan2(y, x))
    )
    data = jnp.asarray(image[~mask], dtype=jnp.float32)
    bounds = tuple(jnp.asarray(v, dtype=jnp.float32) for v in bbox)
    lambda_reg = jnp.asarray(strength, dtype=jnp.float32)

    @jax.jit
    def prepare(query_x, query_y, bounds):
        # Pass rays as inputs so the compiler cannot precompute their deflection.
        betas = physical.deflection(x=query_x, y=query_y)
        return simulator.precompute_operator_data(*bounds, _betas_sub=betas)

    @jax.jit
    def precondition(noise, bounds, lambda_reg):
        return simulator.build_block_diag_preconditioner(
            noise, *bounds, lambda_reg, regularizer, block_size=block_size
        )

    @jax.jit
    def solve(curvature, rhs, preconditioner):
        return pcg_solve(curvature, rhs, preconditioner, rtol=rtol, max_iter=max_iter)

    mapping, mapping_time = _measure(
        prepare, (*simulator.psf_response.ray_coordinates(config), bounds), repeats, device
    )
    # Operator metadata is static for applications, even when emitted by JIT.
    mapping = mapping._replace(
        n_source=int(mapping.n_source),
        nsub=int(mapping.nsub),
        n_render=int(mapping.n_render),
    )
    factors, precondition_time = _measure(
        precondition, (noise, bounds, lambda_reg), repeats, device
    )
    reg_data = regularizer.make_reg_data(*bounds)
    curvature = simulator.build_source_curvature_operator(
        noise, *bounds, lambda_reg, reg_data, op_data=mapping
    )
    rhs, rhs_seconds = _timed(
        lambda: simulator.build_rhs(data, noise, *bounds, op_data=mapping)
    )
    (values, status), solve_time = _measure(
        solve, (curvature, rhs, factors), repeats, device
    )
    rhs_norm = float(jnp.linalg.norm(rhs))
    true_relative_residual = float(jnp.linalg.norm(curvature.matvec(values) - rhs))
    true_relative_residual /= rhs_norm
    if not bool(status.converged) or bool(status.failed) or true_relative_residual > rtol:
        raise RuntimeError(
            f"PCG failed its true residual check ({true_relative_residual:.4g}): {status}"
        )
    stencil_width = 16 if interpolation == "cubic_hermite" else 4
    stencil_shape = (mapping.n_render * nsub**2, stencil_width)
    assert mapping.weights.shape == mapping.indices.shape == stencil_shape
    assert values.shape == (mapping.n_source,) == (n**2,)
    assert factors[0].shape == ((n // block_size) ** 2, block_size**2, block_size**2)
    return dict(
        environment=dict(
            device=device.device_kind,
            python=platform.python_version(),
            jax=jax.__version__,
            jaxlib=jaxlib.__version__,
            numpy=np.__version__,
            dtype=values.dtype.name,
            matmul_precision=jax.config.jax_default_matmul_precision,
        ),
        configuration=dict(
            source_interpolation=interpolation,
            npix=npix,
            n=n,
            dpix=dpix,
            nsub=nsub,
            source_bbox=bbox,
            active_pixels=simulator.n_active,
            render_pixels=mapping.n_render,
            mask_inner_outer_radius=(0.2, 1.8),
            psf=psf.tolist(),
            lens=mass,
            regularization=source.regularization_type,
            lambda_reg=strength,
            block_size=block_size,
            max_iter=max_iter,
            rtol=rtol,
            atol=1e-12,
        ),
        mapping=mapping_time,
        preconditioner=precondition_time,
        rhs_first_seconds=rhs_seconds,
        solve=solve_time,
        reconstruction=dict(
            converged=bool(status.converged),
            failed=bool(status.failed),
            iterations=int(status.n_iter),
            true_relative_residual=true_relative_residual,
            recursive_relative_residual=float(status.residual_norm) / rhs_norm,
        ),
        storage=dict(
            source_nodes=mapping.n_source,
            stencil_shape=stencil_shape,
            stencil_bytes=mapping.weights.nbytes + mapping.indices.nbytes,
            block_factor_shape=factors[0].shape,
            block_factor_bytes=factors[0].nbytes,
            hypothetical_dense_subpixel_bytes=stencil_shape[0] * n**2 * values.itemsize,
            allocator_peak_bytes=device.memory_stats()["peak_bytes_in_use"],
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("interpolation", choices=("bilinear", "cubic_hermite"))
    parser.add_argument("--repeats", type=int, default=10)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    with jax.experimental.enable_x64(False), jax.default_matmul_precision("highest"):
        report = run(args.interpolation, args.repeats)
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
