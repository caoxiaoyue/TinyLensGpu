"""Audit full-matrix allocations in an isolated, generated L-BFGS-B operator fit."""

import argparse
from dataclasses import asdict
from itertools import combinations
import json
from math import prod
import os
from pathlib import Path
import re
import subprocess
import sys
from tempfile import TemporaryDirectory
from time import perf_counter


_HLO_ARRAY = re.compile(r"\b(f\d+|bf16|c\d+|s\d+|u\d+|pred)\[([0-9,]+)\]")


def _fit_worker():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

    import jax
    import jax.numpy as jnp
    import numpy as np

    from examples.potential_correction._lbfgs_resources import measure_lbfgs_fit
    from examples.potential_correction.result_products import json_safe
    from TinyLensGpu.ForwardSimulation import (
        PotentialCorrectionRenderer,
        SimulatorConfig,
    )
    from TinyLensGpu.ObservationModel import (
        LBFGSPotentialCorrection,
        PotentialCorrectionStrengths,
    )
    from TinyLensGpu.PhysicalModel import PhysicalModel, Shear
    from TinyLensGpu.utils.lensing.potential import PotentialGrid

    started = perf_counter()
    with jax.experimental.enable_x64():
        macro = PhysicalModel(
            lens_mass=[Shear(gamma1=0.03, gamma2=0.01)],
            source_light=[],
            lens_light=[],
        )
        psf = np.array([[0.0, 0.05, 0.0], [0.1, 0.7, 0.1], [0.0, 0.05, 0.0]])
        config = SimulatorConfig(0.04, 36, psf, nsub=1)
        query_x, query_y = config.xgrid_sub_1d, config.ygrid_sub_1d
        spacing = 0.08
        base = PotentialGrid.from_queries(query_x, query_y, spacing=spacing)
        # Isolated support-only coefficients grow the fitted potential field
        # while leaving the observed image and derivative domain unchanged.
        extra = np.column_stack((100 + 3 * np.arange(4096), np.full(4096, 100)))
        grid = PotentialGrid.from_queries(
            query_x,
            query_y,
            spacing=spacing,
            node_indices=np.vstack((base.node_indices, extra)),
        )
        renderer = PotentialCorrectionRenderer(
            macro,
            config,
            64,
            (-0.84, 0.84, -0.84, 0.84),
            potential_grid=grid,
            source_interpolation="cubic_hermite",
        )
        sx, sy = jnp.meshgrid(renderer.source_x_axis, renderer.source_y_axis)
        source = (
            jnp.exp(-((sx - 0.07) ** 2 / 0.18 + (sy + 0.09) ** 2 / 0.12)) - 0.09
        ).ravel()
        potential = 0.0015 * jnp.exp(
            -((grid.x - 0.16) ** 2 + (grid.y + 0.08) ** 2) / 0.05
        )
        image = renderer.render(source, potential).model_image
        image += np.random.default_rng(946).normal(
            0.0, 0.012, size=renderer.image_shape
        )
        problem = LBFGSPotentialCorrection(
            renderer,
            image,
            jnp.full(renderer.image_shape, 0.12),
            0.82 * source + 0.02,
            source_regularization="zero-order",
        )
        strengths = PotentialCorrectionStrengths(10.0, 30.0, 350000.0)
        jax.block_until_ready(
            (problem.initial_values, problem.noise_logdet, renderer.psf_fft)
        )
        preparation_seconds = perf_counter() - started
        result, resources = measure_lbfgs_fit(
            problem, strengths, backend="operator", block_size=32
        )

    configuration = dict(result.configuration)
    configuration.pop("timings")
    # The fixed generated partition can be recovered from its layout and sizes.
    configuration.pop("block_indices")
    record = {
        "fit": {
            "backend": result.backend,
            "success": result.success,
            "failure_reason": result.failure_reason,
            "iterations": result.optimizer_info.iterations,
            "gn_stationarity": result.optimizer_info.gn_stationarity,
            "stationarity_solver": result.optimizer_info.diagnostics.get(
                "stationarity_solver"
            ),
            "objective": result.objective,
            "log_evidence": result.log_evidence,
            "determinant_method": result.determinant_method,
        },
        "configuration": configuration,
        "strengths": asdict(strengths),
        "resources": {
            **resources,
            "synchronized_preparation_seconds": preparation_seconds,
        },
        "live_shapes": sorted({tuple(array.shape) for array in jax.live_arrays()}),
    }
    print(json.dumps(json_safe(record), allow_nan=False))


def _audit_allocations(dump_directory, configuration, live_shapes):
    source = configuration["n_source"]
    potential = configuration["n_potential"]
    joint = source + potential
    data = configuration["n_data"]
    forbidden = {
        "source_prior_or_curvature": (source, source),
        "potential_prior_or_curvature": (potential, potential),
        "joint_prior_or_curvature": (joint, joint),
        "cross_curvature": (source, potential),
        "source_response": (source, data),
        "potential_response": (potential, data),
        "joint_response": (joint, data),
        "data_covariance": (data, data),
    }
    by_axes = {tuple(sorted(shape)): name for name, shape in forbidden.items()}
    by_elements = {prod(shape): name for name, shape in forbidden.items()}
    violations = {}
    bounded_shapes = set()
    width = max(configuration["block_sizes"])

    def check_shape(shape, location):
        for pair in combinations(shape, 2):
            name = by_axes.get(tuple(sorted(pair)))
            if name is not None:
                violations.setdefault((name, shape), location)
        # Also catch a flattened or reshaped complete buffer.
        name = by_elements.get(prod(shape))
        if name is not None:
            violations.setdefault((name, shape), location)

    modules = sorted(dump_directory.glob("*before_optimizations.txt"))
    for path in modules:
        shapes = set(_HLO_ARRAY.findall(path.read_text()))
        for dtype, dimensions in shapes:
            shape = tuple(map(int, dimensions.split(",")))
            check_shape(shape, path.name)
            if dtype.startswith("f") and shape[-2:] == (width, width):
                bounded_shapes.add(shape)

    for shape in live_shapes:
        check_shape(tuple(shape), "live device arrays")

    return {
        "method": "XLA HLO array shapes before optimization and synchronized live device array shapes",
        "compiled_modules": len(modules),
        "bounded_block_shapes": sorted(bounded_shapes),
        "prohibited_matrix_shapes": forbidden,
        "violations": [
            {"matrix": name, "shape": shape, "location": location}
            for (name, shape), location in sorted(violations.items())
        ],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker:
        _fit_worker()
        return

    with TemporaryDirectory(prefix="lbfgs-operator-hlo-") as directory:
        flags = os.environ.get("XLA_FLAGS", "")
        completed = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--worker"],
            capture_output=True,
            text=True,
            env={
                **os.environ,
                "XLA_FLAGS": f"{flags} --xla_dump_to={directory} --xla_dump_hlo_as_text",
                "JAX_ENABLE_COMPILATION_CACHE": "false",
                "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
            },
        )
        if completed.returncode:
            sys.stderr.write(completed.stderr + completed.stdout)
            raise SystemExit(completed.returncode)
        record = json.loads(completed.stdout)
        audit = _audit_allocations(
            Path(directory), record["configuration"], record["live_shapes"]
        )

    fit = record["fit"]
    solver = fit["stationarity_solver"] or {}
    record["allocation_audit"] = audit
    record["passed"] = bool(
        fit["success"]
        and fit["iterations"] > 0
        and fit["log_evidence"] is not None
        and solver.get("converged")
        and solver["iterations"] > 0
        and audit["compiled_modules"]
        and audit["bounded_block_shapes"]
        and not audit["violations"]
    )
    print(json.dumps(record, indent=2, allow_nan=False))
    if not record["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
