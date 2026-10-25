"""Validate PNPG on the unmodified 20-MGE fit_joint example.

Run in the tinylens_gpu environment after the example's sim_data.py.
Executes only example setup, stopping before nested sampling. No model,
regularization, precision, or convergence implementation is changed.
"""

import argparse
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

ROOT = Path(__file__).resolve().parents[2]
# The environment's editable install may point at a different checkout.
sys.path.insert(0, str(ROOT))
EXAMPLE = ROOT / "examples/pix_src_demo_operator/simple/pix_src_lens_light_mge/fit_joint.py"
OUT = Path(__file__).resolve().parent / "output"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--solver", choices=("pnpg",), required=True)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--rtol", type=float)
    parser.add_argument("--log-lambda", type=float, nargs="+", default=[0.0, -8.0, -6.0])
    parser.add_argument("--nearby-points", type=int, default=0)
    parser.add_argument("--timing-repeats", type=int, default=3)
    args = parser.parse_args()
    OUT.mkdir(exist_ok=True)

    source, marker, _ = EXAMPLE.read_text().partition(
        'print("[5] Running Nautilus sampler ...")'
    )
    if not marker:
        raise RuntimeError("Example setup boundary changed; refusing to run sampling")
    namespace = {"__file__": str(EXAMPLE), "__name__": "solver_comparison_setup"}
    exec(compile(source, str(EXAMPLE), "exec"), namespace)

    import jax
    import jax.numpy as jnp
    import numpy as np
    import TinyLensGpu

    if Path(TinyLensGpu.__file__).resolve().parent != ROOT / "TinyLensGpu":
        raise RuntimeError("Comparison imported a different TinyLensGpu checkout")

    model = namespace["prob_model"]
    names = namespace["param_names"]
    model.solver_type = args.solver
    setattr(model, f"{args.solver}_max_iter", args.iterations)
    if args.rtol is not None:
        setattr(model, f"{args.solver}_rtol", args.rtol)
    tolerance = getattr(model, f"{args.solver}_rtol")
    initial = np.asarray(model.get_values(), dtype=np.float32)
    print("DYNAMIC_ORDER", names, flush=True)
    evaluate = jax.jit(lambda theta: model.evaluate(theta))
    rows = []
    cases = []
    for log_lambda in args.log_lambda:
        theta = initial.copy()
        theta[names.index("log_lambda_reg")] = log_lambda
        cases.append((f"lambda{log_lambda:g}", theta))
    if args.nearby_points:
        rng = np.random.default_rng(24)
        scales = np.asarray([
            0.1 if name == "log_lambda_reg" else
            0.001 if "center_" in name else 0.003
            for name in names
        ], dtype=np.float32)
        center = cases[0][1]
        for index in range(args.nearby_points):
            theta = center + rng.normal(size=initial.size).astype(np.float32) * scales
            cases.append((f"nearby{index + 1}", theta))
    suffix = f"_nearby{args.nearby_points}" if args.nearby_points else ""
    for point, theta in cases:
        log_lambda = float(theta[names.index("log_lambda_reg")])
        theta = jnp.asarray(theta)
        started = time.perf_counter()
        result = jax.block_until_ready(evaluate(theta))
        first_seconds = time.perf_counter() - started
        timings = []
        for _ in range(args.timing_repeats):
            started = time.perf_counter()
            result = jax.block_until_ready(evaluate(theta))
            timings.append(time.perf_counter() - started)
        info = {key: np.asarray(value).item() for key, value in result.solver_info._asdict().items()}
        fitted = ~np.asarray(namespace["mask"], dtype=bool)
        residual = (np.asarray(namespace["image_data"]) - np.asarray(result.model_image)) / np.asarray(namespace["noise_map"])
        converged = bool(result.solver_info.converged)
        row = dict(
            solver=args.solver,
            iterations=args.iterations,
            rtol=tolerance,
            log_lambda=log_lambda,
            point=point,
            parameters=np.asarray(theta).tolist(),
            solver_info=info,
            log_evidence=float(result.log_evidence),
            chi2_per_data=float(np.mean(residual[fitted] ** 2)) if converged else None,
            source_min=float(result.source_pixels.min()) if converged else None,
            lens_min=float(result.lens_amplitudes.min()) if converged else None,
            first_seconds=first_seconds,
            warmed_seconds_median=float(np.median(timings)),
        )
        print("RESULT", json.dumps(row), flush=True)
        rows.append(row)
        np.savez_compressed(
            OUT / f"{args.solver}_{args.iterations}_{tolerance:g}_{point}.npz",
            theta=np.asarray(theta), model_image=np.asarray(result.model_image),
            source=np.asarray(result.source_pixels), lens=np.asarray(result.lens_amplitudes),
            source_bbox=np.asarray(result.source_bbox),
        )
        report = dict(
            devices=[str(device) for device in jax.devices()],
            jax_version=jax.__version__,
            x64_enabled=jax.config.jax_enable_x64,
            example=str(EXAMPLE.relative_to(ROOT)),
            package_path=str(Path(TinyLensGpu.__file__).resolve()),
            source_n=model.sim_obj.source_n,
            n_lens_light=model.n_lens_light,
            nsub=model.sim_obj.nsub,
            data_seed=42,
            image_shape=list(model.image_data.shape),
            fitted_pixels=int(fitted.sum()),
            source_bbox_padding=0.2,
            lens_light_regularization=model.lens_light_regularization,
            initial_parameters=initial.tolist(),
            nearby_seed=24,
            timing_repeats=args.timing_repeats,
            dynamic_names=names,
            rows=rows,
        )
        (OUT / f"{args.solver}_{args.iterations}_{tolerance:g}{suffix}.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
