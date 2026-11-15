"""Nautilus three-strength posterior with explicit precision and a median fit.

Float32 candidates that fail their declared residual/factorization checks are
evaluated in Float64, rather than removing valid hyperprior volume. Every
fallback and both precision scores are retained. This is a mixed-precision
experiment, not a claim of reliable pure-Float32 evidence over the broad prior.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import jax.scipy as jsp  # noqa: E402
import numpy as np  # noqa: E402
from nautilus import Sampler  # noqa: E402
from scipy.special import logsumexp  # noqa: E402

from optimize_single_step_three_strengths import Objective  # noqa: E402
from plot_three_strength_posterior import quantile  # noqa: E402
from sample_single_step_three_strengths import prepare_data, NAMES  # noqa: E402
from TinyLensGpu.ForwardSimulation import (  # noqa: E402
    PotentialCorrectionRenderer,
    SimulatorConfig,
)  # noqa: E402
from TinyLensGpu.ObservationModel import SingleStepPotentialCorrection  # noqa: E402
from TinyLensGpu.PhysicalModel import PhysicalModel, SIS, Shear  # noqa: E402
from examples.potential_correction.mock import (  # noqa: E402
    MockConfig,
    generate_mock_pair,
)  # noqa: E402
from examples.potential_correction.shared.result_products import (  # noqa: E402
    export_result,
    json_safe,
    write_json,
)  # noqa: E402


def kernel(x, gram, rhs, design, data, noise, source, roughness, solve_rtol):
    with jax.default_matmul_precision("highest"):
        ls, l4, l0 = 10.0**x
        ns, npot = source.shape[0], roughness.shape[0]
        q = jnp.zeros_like(gram).at[:ns, :ns].set(ls * source)
        q = q.at[ns:, ns:].set(l4 * roughness + l0 * jnp.eye(npot, dtype=gram.dtype))
        h = gram + q
        qc, hc = jnp.linalg.cholesky(q), jnp.linalg.cholesky(h)
        values = jsp.linalg.cho_solve((hc, True), rhs)
        residual = (data - design @ values) / noise
        logz = -0.5 * (
            residual @ residual
            + values @ q @ values
            + 2 * jnp.log(jnp.diag(hc)).sum()
            - 2 * jnp.log(jnp.diag(qc)).sum()
            + 2 * jnp.log(noise).sum()
            + data.size * jnp.log(2 * jnp.pi)
        )
        relative = jnp.linalg.norm(h @ values - rhs) / jnp.linalg.norm(rhs)
        valid = jnp.isfinite(logz) & jnp.isfinite(relative) & (relative <= solve_rtol)
        return jnp.stack([jnp.where(valid, logz, -jnp.inf), relative])


batch_kernel = jax.jit(jax.vmap(kernel, in_axes=(0,) + (None,) * 8))


class Evidence:
    def __init__(self, problem, batch_size, tolerance):
        self.problem, self.batch_size = problem, batch_size
        design = problem.dense_mapping()
        problem.dense_prior(1.0, 1.0).block_until_ready()
        gram, rhs = prepare_data(design, problem.data_active, problem.noise_active)
        self.arguments = (
            gram,
            rhs,
            design,
            problem.data_active,
            problem.noise_active,
            problem._dense_source_prior,
            problem._dense_potential_roughness,
            jnp.asarray(tolerance, dtype=design.dtype),
        )

    def compute(self, points):
        points = np.atleast_2d(points)
        outputs = []
        for start in range(0, len(points), self.batch_size):
            panel = points[start : start + self.batch_size]
            padded = np.repeat(panel[-1:], self.batch_size, axis=0)
            padded[: len(panel)] = panel
            result = batch_kernel(
                jnp.asarray(padded, dtype=self.problem.dtype), *self.arguments
            )
            assert result.dtype == self.problem.dtype
            outputs.append(np.asarray(result)[: len(panel)])
        return np.concatenate(outputs)


def precision_policy(dtype):
    """Metadata-only record of the unchanged numerical precision policy."""
    return dict(
        primary_dtype=dtype,
        primary_solve_rtol=1e-5 if dtype == "float32" else 1e-8,
        reference_dtype="float64",
        reference_solve_rtol=1e-8,
        matmul_precision="highest",
        fallback_dtype="float64" if dtype == "float32" else None,
    )


def resume_problems(existing, cfg, settings):
    """State/provenance checks a persisted run must pass before any resume write."""
    problems = []
    for key, current in (
        ("scientific_configuration", cfg),
        ("sampler_settings", settings),
        ("parameter_order", NAMES),
    ):
        if existing.get(key) != current:
            problems.append(f"manifest {key} does not match the current run")
    expected = precision_policy(settings["dtype"])
    saved = existing.get("precision_policy")
    if saved is None:
        # Manifests written before the explicit field was introduced encode a
        # historical policy through their saved dtype. These literal values
        # deliberately stay independent of the current precision_policy().
        saved_dtype = existing.get("sampler_settings", {}).get("dtype")
        saved = {
            "primary_dtype": saved_dtype,
            "primary_solve_rtol": {"float32": 1e-5, "float64": 1e-8}.get(saved_dtype),
            "reference_dtype": "float64",
            "reference_solve_rtol": 1e-8,
            "matmul_precision": "highest",
            "fallback_dtype": "float64" if saved_dtype == "float32" else None,
        }
    if saved != expected:
        problems.append(
            f"manifest precision policy {saved} does not match this run {expected}"
        )
    return problems


def problem_for(dtype, observation, cfg, frozen):
    def cast(x):
        return jnp.asarray(x, dtype=dtype)

    m = cfg["mock"]
    macro = PhysicalModel(
        lens_mass=[
            SIS(
                theta_E=cast(m["theta_E"]),
                center_x=cast(m["macro_center_x"]),
                center_y=cast(m["macro_center_y"]),
            ),
            Shear(gamma1=cast(m["gamma1"]), gamma2=cast(m["gamma2"])),
        ],
        source_light=[],
        lens_light=[],
    )
    renderer = PotentialCorrectionRenderer(
        macro,
        SimulatorConfig(
            m["dpix"],
            m["npix"],
            cast(observation.psf_kernel),
            cfg["inversion_nsub"],
            observation.mask,
        ),
        cfg["source_n"],
        tuple(cfg["source_bbox"]),
        potential_spacing=cfg["potential_spacing"],
        dtype=dtype,
    )
    gradient = (
        observation.source_truth.gradient
        if frozen is None
        else lambda bx, by: tuple(cast(g) for g in frozen)
    )
    return SingleStepPotentialCorrection(
        renderer,
        cast(observation.image_data),
        cast(observation.noise_map),
        gradient,
        lambda_amplitude=cfg["lambda_amplitude"],
        solve_rtol=1e-5 if dtype == "float32" else 1e-8,
    )


def median_reference(out, cfg, reference=None, obs=None):
    """Check Float32 median fields against a public Float64 fit at the same point."""
    with jax.experimental.enable_x64():
        if reference is None:
            obs = generate_mock_pair(MockConfig.from_dict(cfg["mock"])).subhalo
            reference = problem_for("float64", obs, cfg, None)
        with np.load(out / "posterior.npz") as a:
            w = a["weights"] / a["weights"].sum()
            median = np.array(
                [quantile(a["log10_strengths"][:, i], w, [0.5])[0] for i in range(3)]
            )
        result = Objective(reference, out / "median-reference.jsonl").fit(median)
        assert result.success
        export_result(
            result,
            obs,
            out / "posterior-median-float64",
            configuration=dict(cfg, selected_log10_strengths=median.tolist()),
            make_plots=False,
        )
        with np.load(out / "posterior-median/reconstruction.npz") as a:
            np.testing.assert_array_equal(
                a["potential_node_indices"], result.potential_grid.node_indices
            )
            valid = a["derivative_valid"]
            fields = {}
            for name in ("source_values", "potential_values", "convergence"):
                x, y = np.asarray(getattr(result, name), dtype=float), np.asarray(
                    a[name], dtype=float
                )
                if name == "convergence":
                    x, y = x[valid], y[valid]
                fields[name] = dict(
                    max_absolute=np.max(abs(y - x)),
                    relative_l2=np.linalg.norm(y - x) / np.linalg.norm(x),
                )
        primary = json.loads((out / "posterior-median/result.json").read_text())
        write_json(
            out / "median-reconstruction-precision.json",
            dict(
                fields=fields,
                float32_minus_float64_log_evidence=primary["log_evidence"]
                - float(result.log_evidence),
                float64_summary=json.loads(
                    (out / "posterior-median-float64/result.json").read_text()
                )["summary"],
            ),
        )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    p.add_argument("--n-live", type=int, default=300)
    p.add_argument("--n-eff", type=int, default=5000)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--seed", type=int, default=271828)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--validate-only", action="store_true")
    p.add_argument("--median-reference-only", action="store_true")
    args = p.parse_args()
    out = args.output.resolve()
    persisted = args.resume or args.median_reference_only
    manifest_path = out / "manifest.json"
    if persisted:
        if not out.is_dir():
            p.error(f"existing run directory required for a persisted run: {out}")
        if not manifest_path.is_file():
            p.error(f"persisted run is missing its manifest: {manifest_path}")
        saved = json.loads(manifest_path.read_text())
    else:
        out.mkdir(parents=True)
    if args.median_reference_only:
        # Reconstruct from the frozen persisted inputs, not today's
        # mutable scientific_configuration.json.
        median_reference(out, saved["scientific_configuration"])
        return
    cfg = json.loads(
        (
            ROOT / "examples/potential_correction/mock/single_step/configurations/scientific.json"
        ).read_text()
    )
    bounds = np.array([[-8.0, -1.0], [1.0, 11.0], [0.0, 8.0]])
    settings = dict(
        n_live=args.n_live,
        n_eff=args.n_eff,
        batch_size=args.batch_size,
        seed=args.seed,
        case="subhalo",
        dtype=args.dtype,
        log10_prior_bounds=bounds.tolist(),
        f_live=0.01,
        discard_exploration=True,
    )
    manifest = dict(
        date=datetime.now(timezone.utc).isoformat(),
        scientific_configuration=cfg,
        sampler_settings=settings,
        parameter_order=NAMES,
        target="Full dense conditional Gaussian evidence, with independent log-uniform hyperpriors",
        fallback="Float64 for invalid Float32 evidence candidates",
        driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        precision_policy=precision_policy(args.dtype),
    )
    if args.resume:
        problems = resume_problems(saved, cfg, settings)
        if problems:
            raise ValueError("Refusing --resume: " + "; ".join(problems))
        # Legacy manifests gain the explicit policy only after every check
        # passes; saved configuration and historical driver hashes are kept.
        saved["precision_policy"] = manifest["precision_policy"]
        saved.setdefault("resumptions", []).append(
            dict(
                driver_sha256=manifest["driver_sha256"],
                reason="Resume continued from the persisted sampler checkpoint",
            )
        )
        write_json(manifest_path, saved)
    else:
        write_json(manifest_path, manifest)
    with jax.experimental.enable_x64():
        assert jax.devices()[0].platform == "gpu"
        pair = generate_mock_pair(MockConfig.from_dict(cfg["mock"]))
        obs = pair.subhalo
        reference = problem_for("float64", obs, cfg, None)
        frozen = reference.source_gradient(*reference.macro_mapping)
        problem = (
            reference
            if args.dtype == "float64"
            else problem_for("float32", obs, cfg, frozen)
        )
        for name in ("node_indices", "core_indices"):
            np.testing.assert_array_equal(
                getattr(problem.renderer.potential_grid, name),
                getattr(reference.renderer.potential_grid, name),
            )
        ref = Evidence(reference, args.batch_size, 1e-8)
        primary = (
            ref if args.dtype == "float64" else Evidence(problem, args.batch_size, 1e-5)
        )
        old = np.load(
            ROOT
            / ".scratch/potential-correction/research/three-strength-nautilus/posterior.npz"
        )
        weights = old["weights"] / old["weights"].sum()
        reference_median = np.array(
            [
                quantile(old["log10_strengths"][:, i], weights, [0.5])[0]
                for i in range(3)
            ]
        )
        rng = np.random.default_rng(12345)
        checks = np.vstack(
            [
                reference_median,
                old["log10_strengths"][rng.choice(len(weights), 32, p=weights)],
                [-4, 10, 1],
                [-4, 2, 7],
            ]
        )
        a, b = primary.compute(checks), ref.compute(checks)
        posterior_error = a[:33, 0].astype(float) - b[:33, 0]
        assert (
            np.all(np.isfinite(posterior_error)) and np.max(abs(posterior_error)) < 0.1
        ), posterior_error
        public = Objective(problem, out / "public-validation.jsonl").fit(
            reference_median
        )
        assert public.success
        assert abs(float(public.log_evidence) - float(a[0, 0])) < 0.1
        write_json(
            out / ("resume-validation.json" if args.resume else "validation.json"),
            dict(
                coordinates=checks,
                primary=a,
                float64=b,
                posterior_region_max_abs_delta_logz=np.max(abs(posterior_error)),
                public_fit_logz=float(public.log_evidence),
                primary_dtype=str(primary.arguments[0].dtype),
                requested_solve_rtol=1e-5 if args.dtype == "float32" else 1e-8,
            ),
        )
        print(
            json.dumps(dict(validation_max_error=float(np.max(abs(posterior_error))))),
            flush=True,
        )
        if args.validate_only:
            return
        calls, fallbacks = 0, 0

        def likelihood(points):
            nonlocal calls, fallbacks
            x = np.atleast_2d(points)
            result = primary.compute(x).astype(float)
            bad = ~np.isfinite(result[:, 0])
            original = result.copy()
            if bad.any() and args.dtype == "float32":
                result[bad] = ref.compute(x[bad])
                fallbacks += int(bad.sum())
            calls += len(x)
            with (out / "likelihood-evaluations.jsonl").open("a") as stream:
                for point, row, raw, fallback in zip(x, result, original, bad):
                    stream.write(
                        json.dumps(
                            dict(
                                log10_strengths=point.tolist(),
                                log_evidence=(
                                    float(row[0]) if np.isfinite(row[0]) else None
                                ),
                                primary_log_evidence=(
                                    float(raw[0]) if np.isfinite(raw[0]) else None
                                ),
                                relative_residual=(
                                    float(row[1]) if np.isfinite(row[1]) else None
                                ),
                                fallback_float64=bool(
                                    fallback and args.dtype == "float32"
                                ),
                            )
                        )
                        + "\n"
                    )
            return result[:, 0]

        sampler = Sampler(
            lambda u: bounds[:, 0] + np.asarray(u) * np.diff(bounds).ravel(),
            likelihood,
            n_dim=3,
            n_live=args.n_live,
            n_batch=args.batch_size,
            vectorized=True,
            seed=args.seed,
            filepath=out / "checkpoint.h5",
            resume=args.resume,
        )
        started = time.perf_counter()
        complete = False
        while not complete:
            complete = bool(
                sampler.run(
                    n_eff=args.n_eff,
                    f_live=0.01,
                    discard_exploration=True,
                    timeout=40,
                    verbose=False,
                )
            )
            sampler.write(out / "checkpoint.h5", overwrite=True)
            state = dict(
                complete=complete,
                n_live=args.n_live,
                n_like=sampler.n_like,
                explored=sampler.explored,
                f_live=None if sampler.f_live is None else float(sampler.f_live),
                n_eff=float(sampler.n_eff),
                hyperprior_integrated_log_evidence=float(sampler.log_z),
                calls_this_invocation=calls,
                fallbacks_this_invocation=fallbacks,
                elapsed_seconds_this_invocation=time.perf_counter() - started,
            )
            write_json(out / "progress.json", state)
            print(json.dumps(json_safe(state), allow_nan=False), flush=True)
        points, logw, loglike = sampler.posterior(equal_weight=False)
        weights = np.exp(logw - logsumexp(logw))
        np.savez_compressed(
            out / "posterior.npz",
            log10_strengths=points,
            strengths=10.0**points,
            log_weights=logw,
            weights=weights,
            conditional_log_evidence=loglike,
            parameter_order=np.asarray(NAMES),
        )
        state.update(
            posterior_points=len(points),
            weighted_kish_effective_size=float(1 / (weights @ weights)),
        )
        write_json(out / "completion.json", state)
        median = np.array([quantile(points[:, i], weights, [0.5])[0] for i in range(3)])
        fit = Objective(problem, out / "median-fit.jsonl").fit(median)
        assert fit.success, fit.failure_reason
        export_result(
            fit,
            obs,
            out / "posterior-median",
            configuration=dict(cfg, selected_log10_strengths=median.tolist()),
            make_plots=False,
        )
        # Independent Float64 check of every retained weighted posterior point.
        check64 = ref.compute(points)[:, 0].astype(float)
        assert np.all(np.isfinite(check64))
        delta = loglike - check64
        adjusted = weights * np.exp(-delta - np.max(-delta))
        adjusted /= adjusted.sum()
        write_json(
            out / "posterior-precision-audit.json",
            dict(
                median_log10_strengths=median,
                median_strengths=10.0**median,
                maximum_abs_delta_logz=np.max(abs(delta)),
                weighted_mean_delta_logz=weights @ delta,
                weighted_rms_delta_logz=np.sqrt(weights @ (delta**2)),
                float64_reweighted_kish_size=1 / (adjusted @ adjusted),
                reweighting_total_variation=0.5 * np.sum(abs(weights - adjusted)),
                float64_reweighted_median_strengths=[
                    10.0 ** quantile(points[:, i], adjusted, [0.5])[0] for i in range(3)
                ],
            ),
        )
        print(
            json.dumps(dict(completed=True, median_strengths=(10.0**median).tolist())),
            flush=True,
        )
        median_reference(out, cfg, reference, obs)


if __name__ == "__main__":
    main()
