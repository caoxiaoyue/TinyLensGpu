"""Nautilus hyperposterior for the unchanged scientific single-step model.

The inner likelihood is the full conditional Gaussian evidence, with source
and potential fields integrated out. Cache the strength-independent data
Gram matrix; keep full dense Cholesky determinants and the actual image
residual. Validate against public fits before sampling. No block, eigenvalue,
Laplace, or surrogate evidence approximation is introduced.
"""

import argparse
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
from nautilus import Sampler

from optimize_single_step_three_strengths import Objective
from TinyLensGpu.ForwardSimulation import PotentialCorrectionRenderer, SimulatorConfig
from TinyLensGpu.ObservationModel import SingleStepPotentialCorrection
from examples.potential_correction.mock import MockConfig, generate_mock_pair
from examples.potential_correction.result_products import json_safe, write_json


NAMES = ["lambda_source", "lambda_potential", "lambda_amplitude"]


@jax.jit
def prepare_data(design, data, noise):
    with jax.default_matmul_precision("highest"):
        weighted = design / noise[:, None]
        return weighted.T @ weighted, weighted.T @ (data / noise)


def evidence_kernel(log10_strengths, gram, rhs, design, data, noise, source, potential):
    with jax.default_matmul_precision("highest"):
        ls, lp, la = 10.0**log10_strengths
        ns, npot = source.shape[0], potential.shape[0]
        prior = jnp.zeros_like(gram).at[:ns, :ns].set(ls*source)
        prior = prior.at[ns:, ns:].set(lp*potential + la*jnp.eye(npot, dtype=gram.dtype))
        curvature = gram + prior
        qc = jnp.linalg.cholesky(prior)
        hc = jnp.linalg.cholesky(curvature)
        values = jsp.linalg.cho_solve((hc, True), rhs)
        residual = (data - design @ values) / noise
        chi2 = jnp.vdot(residual, residual)
        penalty = values @ prior @ values
        ldq = 2*jnp.log(jnp.diag(qc)).sum()
        ldh = 2*jnp.log(jnp.diag(hc)).sum()
        logz = -0.5*(chi2 + penalty + ldh - ldq + 2*jnp.log(noise).sum()
                     + data.size*jnp.log(2*jnp.pi))
        relative = jnp.linalg.norm(curvature @ values-rhs) / jnp.linalg.norm(rhs)
        valid = jnp.isfinite(logz) & jnp.isfinite(relative) & (relative <= 1e-8)
        return jnp.stack([jnp.where(valid, logz, -jnp.inf), relative, chi2, penalty, ldq, ldh])


batch_evidence = jax.jit(jax.vmap(evidence_kernel, in_axes=(0, None, None, None, None, None, None, None)))


class CachedEvidence:
    def __init__(self, problem, batch_size, output):
        self.problem, self.batch_size, self.output = problem, batch_size, output
        self.design = problem.dense_mapping()
        problem.dense_prior(1.0, 1.0).block_until_ready()
        self.gram, self.rhs = prepare_data(self.design, problem.data_active, problem.noise_active)
        self.arguments = (self.gram, self.rhs, self.design, problem.data_active,
                          problem.noise_active, problem._dense_source_prior,
                          problem._dense_potential_roughness)
        self.calls = 0
        self.failed = 0
        self.maximum_residual = 0.0
        self.best_logz = -np.inf
        self.best_x = None

    def compute(self, points):
        points = np.atleast_2d(np.asarray(points, dtype=np.float64))
        result = []
        for start in range(0, len(points), self.batch_size):
            panel = points[start:start+self.batch_size]
            padded = np.repeat(panel[-1:], self.batch_size, axis=0)
            padded[:len(panel)] = panel
            result.append(np.asarray(batch_evidence(jnp.asarray(padded), *self.arguments))[:len(panel)])
        return np.concatenate(result)

    def __call__(self, points):
        coordinates = np.atleast_2d(points)
        outputs = self.compute(coordinates)
        finite = np.isfinite(outputs[:, 0])
        self.calls += len(outputs)
        self.failed += int((~finite).sum())
        if finite.any():
            self.maximum_residual = max(self.maximum_residual, float(outputs[finite, 1].max()))
            index = int(np.argmax(outputs[:, 0]))
            if outputs[index, 0] > self.best_logz:
                self.best_logz = float(outputs[index, 0])
                self.best_x = coordinates[index].tolist()
        with (self.output / "likelihood-evaluations.jsonl").open("a") as stream:
            for x, row in zip(coordinates, outputs):
                stream.write(json.dumps(dict(log10_strengths=np.asarray(x).tolist(),
                    log_evidence=float(row[0]) if np.isfinite(row[0]) else None,
                    relative_residual=float(row[1]) if np.isfinite(row[1]) else None)) + "\n")
        return outputs[:, 0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case", choices=["subhalo", "control"], default="subhalo")
    parser.add_argument("--n-live", type=int, default=300)
    parser.add_argument("--n-eff", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=314159)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=args.resume)
    config_path = ROOT / "examples/potential_correction/scientific_configuration.json"
    config = json.loads(config_path.read_text())
    bounds = np.array([[-8., -1.], [1., 11.], [0., 8.]])
    settings = dict(n_live=args.n_live, n_eff=args.n_eff, batch_size=args.batch_size,
                    seed=args.seed, case=args.case, log10_prior_bounds=bounds.tolist(),
                    f_live=0.01, discard_exploration=True)
    manifest_path = args.output / "manifest.json"
    if args.resume and manifest_path.exists():
        existing = json.loads(manifest_path.read_text())
        assert existing["sampler_settings"] == settings
        existing.setdefault("resumptions", []).append(dict(
            driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            reason="Continue persisted sampler state; numerical target and settings unchanged",
        ))
        write_json(manifest_path, existing)
    else:
        write_json(manifest_path, dict(
            date="2026-10-31", scientific_configuration=config, sampler_settings=settings,
            parameter_order=NAMES,
            target="p(log10 strengths | image) proportional to exp(full conditional Gaussian log evidence); independent bounded uniform log10 hyperpriors",
            fixed="true macro, analytic source gradient, source/potential geometry, data, noise and PSF",
            software=dict(python=platform.python_version(), jax=jax.__version__,
                          numpy=np.__version__, nautilus=importlib.metadata.version("nautilus-sampler")),
            git_commit=subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                      check=True, text=True, capture_output=True).stdout.strip(),
            source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in [
                Path(__file__), config_path,
                ROOT / "TinyLensGpu/ObservationModel/LensImage/potential_correction.py",
                ROOT / "TinyLensGpu/ObservationModel/LensImage/_potential_correction_common.py",
                ROOT / "TinyLensGpu/utils/inversion/potential_regularization.py",
            ]},
        ))
    with jax.experimental.enable_x64():
        assert jax.devices()[0].platform == "gpu"
        mock = MockConfig.from_dict(config["mock"])
        pair = generate_mock_pair(mock)
        observation = getattr(pair, args.case)
        renderer = PotentialCorrectionRenderer(
            pair.control.macro,
            SimulatorConfig(mock.dpix, mock.npix, pair.control.psf_kernel,
                            config["inversion_nsub"], pair.control.mask),
            config["source_n"], tuple(config["source_bbox"]),
            potential_spacing=config["potential_spacing"], dtype="float64",
        )
        problem = SingleStepPotentialCorrection(
            renderer, observation.image_data, observation.noise_map,
            observation.source_truth.gradient, lambda_amplitude=config["lambda_amplitude"],
            solve_rtol=config["solve_rtol"],
        )
        likelihood = CachedEvidence(problem, args.batch_size, args.output)
        # Compare full output terms and true residual gates with public fits.
        reference_path = (ROOT / ".scratch/potential-correction/research/three-strength-single-step"
                          / args.case / "summary.json")
        reference = json.loads(reference_path.read_text())
        mode = np.asarray(reference["selected"]["log10_strengths"])
        original = np.log10([config["fixed_strengths"][args.case][key] for key in NAMES[:2]]
                            + [config["lambda_amplitude"]])
        rng = np.random.default_rng(20261031)
        points = np.vstack([original, mode, mode+rng.normal(0, .4, (6, 3)),
                            bounds[:, 0]+[1, 2, 1], [-4, 10, 1], [-4, 2, 7]])
        if args.resume and (args.output / "validation.json").exists():
            points = points[:2]
        outputs = likelihood.compute(points)
        public = Objective(problem, args.output / "public-validation-evaluations.jsonl")
        validations = []
        for x, output in zip(points, outputs):
            result = public.fit(x)
            assert result.success and np.isfinite(output[0])
            error = abs(float(result.log_evidence)-output[0])
            tolerance = 1e-5 if x[1]-x[2] < 8 else 1e-3
            assert error < tolerance, (x, error, tolerance)
            validations.append(dict(log10_strengths=x.tolist(), log_evidence_error=error,
                                    public_relative_residual=result.solver_info.relative_residual,
                                    cached_relative_residual=float(output[1])))
        warm = []
        for _ in range(3):
            start = time.perf_counter()
            likelihood.compute(np.repeat(mode[None, :], args.batch_size, axis=0))
            warm.append(time.perf_counter()-start)
        validation = dict(checks=validations, maximum_log_evidence_error=max(v["log_evidence_error"] for v in validations),
                          warm_batch_seconds=warm, batch_size=args.batch_size,
                          evidence_evaluations_per_second=args.batch_size/np.median(warm),
                          caching="Only data Gram/rhs and strength-independent prior templates; full dense Q/H Cholesky and actual residual retained")
        write_json(args.output / ("resume-validation.json" if args.resume else "validation.json"), validation)
        print(json.dumps(dict(validation=validation)), flush=True)
        if args.validate_only:
            return

        def prior(unit):
            return bounds[:, 0] + np.asarray(unit)*(bounds[:, 1]-bounds[:, 0])

        sampler = Sampler(prior, likelihood, n_dim=3, n_live=args.n_live,
                          n_batch=args.batch_size, vectorized=True, seed=args.seed,
                          filepath=args.output / "checkpoint.h5", resume=args.resume)
        start = time.perf_counter()
        complete = False
        while not complete:
            complete = bool(sampler.run(n_eff=args.n_eff, f_live=.01, discard_exploration=True,
                                       timeout=40, verbose=False))
            sampler.write(args.output / "checkpoint.h5", overwrite=True)
            state = dict(complete=complete, n_live=sampler.n_live, n_like=sampler.n_like,
                         explored=sampler.explored,
                         f_live=None if sampler.f_live is None else float(sampler.f_live),
                         n_eff=float(sampler.n_eff), hyperprior_integrated_log_evidence=sampler.log_z,
                         failed_likelihoods_this_invocation=likelihood.failed,
                         maximum_relative_residual_this_invocation=likelihood.maximum_residual,
                         best_conditional_log_evidence_this_invocation=likelihood.best_logz,
                         best_log10_strengths_this_invocation=likelihood.best_x,
                         elapsed_seconds_this_invocation=time.perf_counter()-start)
            write_json(args.output / "progress.json", state)
            print(json.dumps(json_safe(state), allow_nan=False), flush=True)
        points, log_weights, log_likelihood = sampler.posterior(equal_weight=False)
        weights = np.exp(log_weights-jsp.special.logsumexp(log_weights))
        np.savez_compressed(args.output / "posterior.npz", log10_strengths=points,
                            strengths=10.0**points, log_weights=log_weights,
                            weights=weights, conditional_log_evidence=log_likelihood,
                            parameter_order=np.asarray(NAMES))
        state.update(posterior_points=len(points), weighted_kish_effective_size=float(1/np.sum(weights**2)))
        write_json(args.output / "completion.json", state)
        print(json.dumps(json_safe(dict(completed=True, **state)), allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
