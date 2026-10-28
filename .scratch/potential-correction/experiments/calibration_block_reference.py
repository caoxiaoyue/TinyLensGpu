"""Independent small dense diagnostic for principal-block layout selection.

This intentionally forms complete matrices; it is neither the production
operator path nor a memory benchmark. All layouts are fixed over the search.
Run from the repository root with PYTHONPATH=. in tinylens_gpu. Example:
  python .scratch/potential-correction/experiments/calibration_block_reference.py --config examples/potential_correction/calibration_configuration.json --output /tmp/block-reference.json --layouts morton128 morton256 incidence24_1024
"""

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import direct

from TinyLensGpu.ForwardSimulation import PotentialCorrectionRenderer
from TinyLensGpu.ObservationModel import SingleStepPotentialCorrection
from examples.potential_correction.calibrate_single_step import (
    spatial_blocks,
    source_incidence_blocks,
    _fit_record,
    _json_safe,
    _truth,
)
from examples.potential_correction.mock import MockConfig, generate_mock_pair


@jax.jit
def gaussian(strengths, design, gram, rs, r4, amplitude, data, noise):
    with jax.default_matmul_precision("highest"):
        q = strengths[0] * rs + strengths[1] * r4 + amplitude
        h = gram + q
        cq, ch = jnp.linalg.cholesky(q), jnp.linalg.cholesky(h)
        rhs = design.T @ (data / noise**2)
        x = jax.scipy.linalg.cho_solve((ch, True), rhs)
        residual = (data - design @ x) / noise
        base = (
            residual @ residual
            + x @ q @ x
            + 2 * jnp.log(noise).sum()
            + len(data) * jnp.log(2 * jnp.pi)
        )
        logq, logh = 2 * jnp.log(jnp.diag(cq)).sum(), 2 * jnp.log(jnp.diag(ch)).sum()
        relative = jnp.linalg.norm(h @ x - rhs) / jnp.linalg.norm(rhs)
        return jnp.stack((-0.5 * (base + logh - logq), base, logq, logh, relative))


@jax.jit
def block_ratio(strengths, data, source, potential, amplitude, valid):
    q = strengths[0] * source + strengths[1] * potential + amplitude
    q = q + jax.vmap(jnp.diag)((~valid).astype(q.dtype))

    def logdet(matrices):
        scale = jnp.sqrt(jnp.diagonal(matrices, axis1=1, axis2=2))
        normalized = matrices / (scale[:, :, None] * scale[:, None, :])
        chol = jnp.linalg.cholesky(normalized)
        return 2 * (
            jnp.log(jnp.diagonal(chol, axis1=1, axis2=2)).sum() + jnp.log(scale).sum()
        )

    return logdet(data + q) - logdet(q)


def principal_arrays(blocks, arrays):
    width = max(map(len, blocks))
    result = [np.zeros((len(blocks), width, width)) for _ in arrays]
    valid = np.zeros((len(blocks), width), dtype=bool)
    for index, block in enumerate(blocks):
        valid[index, : len(block)] = True
        for out, matrix in zip(result, arrays):
            out[index, : len(block), : len(block)] = matrix[np.ix_(block, block)]
    return tuple(jnp.asarray(a) for a in (*result, valid))


def run(config, layouts, *, amplitude, maxfun):
    pair = generate_mock_pair(MockConfig.from_dict(config["mock"]))
    renderer = PotentialCorrectionRenderer(
        pair.control.macro,
        pair.control.sim_config,
        config["source_n"],
        tuple(config["source_bbox"]),
        potential_spacing=config["potential_spacing"],
        dtype="float64",
    )
    prototype = SingleStepPotentialCorrection(
        renderer,
        pair.subhalo.image_data,
        pair.subhalo.noise_map,
        pair.subhalo.source_truth.gradient,
        lambda_amplitude=amplitude,
    )
    design = prototype.dense_mapping()
    amp = prototype.dense_prior(0, 0)
    rs = prototype.dense_prior(1, 0) - amp
    r4 = prototype.dense_prior(0, 1) - amp
    noise = prototype.noise_active
    with jax.default_matmul_precision("highest"):
        gram = (design / noise[:, None]).T @ (design / noise[:, None])
    matrices = [np.asarray(a) for a in (gram, rs, r4, amp)]
    partitions = {}
    for name in layouts:
        if name.startswith("morton"):
            blocks = spatial_blocks(renderer, int(name.removeprefix("morton")))
        else:
            tile, cap = map(int, name.removeprefix("incidence").split("_"))
            blocks = source_incidence_blocks(
                renderer,
                prototype.mapping,
                prototype.gradient_x,
                prototype.gradient_y,
                block_size=cap,
                source_tile=tile,
            )
        partitions[name] = [np.asarray(block).tolist() for block in blocks]
    arrays = {
        name: principal_arrays(blocks, matrices) for name, blocks in partitions.items()
    }
    rows = []
    for case in ("subhalo", "control"):
        obs = getattr(pair, case)
        data = jnp.asarray(obs.image_data).ravel()[renderer.flat_indices]
        problem = SingleStepPotentialCorrection(
            renderer,
            obs.image_data,
            obs.noise_map,
            obs.source_truth.gradient,
            lambda_amplitude=amplitude,
        )
        truth = _truth(renderer, obs)
        for method in ("dense", *partitions):
            calls = []

            def objective(logs):
                strength = jnp.exp(jnp.asarray(logs, dtype=jnp.float64))
                stats = np.asarray(
                    gaussian(strength, design, gram, rs, r4, amp, data, noise)
                )
                value = (
                    stats[0]
                    if method == "dense"
                    else -0.5
                    * (stats[1] + float(block_ratio(strength, *arrays[method])))
                )
                valid = bool(np.isfinite(value) and stats[-1] < 1e-8)
                calls.append(
                    dict(
                        log_strengths=np.asarray(logs).tolist(),
                        log_evidence=float(value) if valid else None,
                        success=valid,
                    )
                )
                return -float(value) if valid else np.inf

            opt = direct(objective, config["log_bounds"], maxfun=maxfun, len_tol=1e-3)
            fit = problem.fit(*np.exp(opt.x))
            row = dict(
                case=case,
                lambda_amplitude=amplitude,
                method=method,
                log_strengths=opt.x.tolist(),
                negative_evidence=float(opt.fun),
                nfev=opt.nfev,
                optimizer_status=opt.status,
                optimizer_message=opt.message,
                complete_selected_fit=_fit_record(fit, truth, obs),
                evaluations=calls,
            )
            rows.append(row)
            print(
                json.dumps(
                    dict(
                        case=case,
                        method=method,
                        logs10=(opt.x / np.log(10)).tolist(),
                        complete_logz=float(fit.log_evidence),
                    )
                ),
                flush=True,
            )
    return dict(
        purpose=__doc__,
        configuration=config,
        lambda_amplitude=amplitude,
        partitions=partitions,
        results=rows,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--layouts", nargs="+", default=["morton128", "morton256", "incidence24_1024"]
    )
    parser.add_argument("--lambda-amplitude", type=float, default=1000.0)
    parser.add_argument("--maxfun", type=int, default=200)
    args = parser.parse_args()
    with jax.experimental.enable_x64():
        result = run(
            json.loads(args.config.read_text()),
            args.layouts,
            amplitude=args.lambda_amplitude,
            maxfun=args.maxfun,
        )
    args.output.write_text(
        json.dumps(_json_safe(result), indent=2, allow_nan=False) + "\n"
    )


if __name__ == "__main__":
    main()
