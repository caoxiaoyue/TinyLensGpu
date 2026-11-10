"""Reproduce the no-noise Euclid mock with the released image-driven prior."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
import subprocess
from time import perf_counter

import numpy as np
from astropy.io import fits
from nautilus import Sampler
from scipy import linalg, sparse, stats
from scipy.optimize import minimize, minimize_scalar

import TinyLensGpu
import caskade as ck
import jax
import jax.numpy as jnp

from TinyLensGpu.Inference import ParamU, StagePosterior
from TinyLensGpu.Inference.build_likelihood import make_likelihood
from TinyLensGpu.Inference.build_prior import make_prior_transformation
from TinyLensGpu.ObservationModel.LensImage._pixelized_regularization import (
    ADAPT_REFERENCE_FLOOR,
    ADAPT_REFERENCE_PERCENTILE,
    HERMITE_COVERAGE_OFFSET,
)
from TinyLensGpu.ObservationModel.LensImage.pixelized_image_model_operator import (
    PixelizedImageProbModelOperator,
)
from TinyLensGpu.PhysicalModel import EPL, SIE, PhysicalModel, PixelizedSourceModel, Shear
from TinyLensGpu.example_support import (
    StageStorageConfig,
    load_stage,
    source_interpolation_metadata,
    stage_from_payload,
)
from TinyLensGpu.utils.misc import weighted_quantile

HERE = Path(__file__).resolve().parent
UPSTREAM = "pix_src_fit_no_noise_upstream"
ARC = "m3_ablation/no_noise_default_overlap_output/m1_psf_arc.npy"
LOG_LAMBDA = "log_lambda_reg"
RHO = "adaptive_reg_rho"
M2_BOUNDS = {LOG_LAMBDA: (-np.log(1e6), np.log(1e6)), RHO: (0.0, 8.0)}
INTERPOLATION = "cubic_hermite"
SEED = 73
OUTER_BATCH = 100
CHUNK = 5
BUDGETS = {"m2": (250, 600), "m3": (180, 600)}


@dataclass(frozen=True)
class MockInputs:
    image: np.ndarray
    noise: np.ndarray
    psf: np.ndarray
    mask: np.ndarray
    adapt_image: np.ndarray
    source_pixels: np.ndarray
    bbox: tuple[float, float, float, float]
    n: int
    dpix: float
    nsub: int
    stage_a: StagePosterior
    stage_m1: StagePosterior
    m1_point: dict
    position_likelihood: dict
    injected_mass: dict
    hashes: dict


def load_mock(archive: Path) -> MockInputs:
    """Read the verified native archive, including its original SIE producer."""
    archive = Path(archive)
    manifest = json.loads((HERE / "euclid_mock_inputs.json").read_text())
    hashes = manifest["sha256"]
    for name, expected in hashes.items():
        actual = sha256((archive / name).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"Archive input identity differs for {name}: {actual}")

    observation = archive / "data_no_noise"
    image, noise, psf = (
        fits.getdata(observation / f"{name}.fits").astype(np.float32)
        for name in ("image", "noise", "psf")
    )
    if fits.getheader(observation / "image.fits")["NOISEADD"]:
        raise ValueError("Expected the no-noise observation with archived RMS weights.")
    storage = StageStorageConfig(archive / UPSTREAM, include_schema=True)
    payload_a = load_stage(storage, "a")
    payload_m1 = load_stage(storage, "m1")
    settings = payload_m1["cache_metadata"]["settings"]
    dpix, nsub = float(settings["dpix"]), int(settings["nsub_pix"])
    feature = fits.getdata(archive / UPSTREAM / "feature_mask.fits").astype(bool)
    height, width = image.shape
    y = (np.arange(height) - (height - 1) / 2) * dpix
    x = (np.arange(width) - (width - 1) / 2) * dpix
    mask = feature | (y[:, None] ** 2 + x[None, :] ** 2 > settings["mask_radius"] ** 2)
    grid = json.loads((archive / UPSTREAM / "source_grid_spec.json").read_text())
    m1 = payload_m1["extra"]["s1"]
    if (
        payload_m1["extra"]["source_grid_spec"] != grid
        or grid["nsrc"] != m1["n"]
        or not np.allclose(grid["pixel_center_bbox"], m1["source_bbox"], atol=1e-7)
    ):
        raise ValueError("M1 source pixels do not use the archived source grid.")
    adapt_image = np.load(archive / ARC)
    if (
        image.shape != (200, 200) or noise.shape != image.shape
        or mask.shape != image.shape or adapt_image.shape != image.shape
        or psf.shape != (21, 21) or grid["nsrc"] != 80 or np.count_nonzero(~mask) != 1339
        or not all(np.isfinite(values).all() for values in (image, noise, psf, adapt_image))
        or np.any(noise <= 0) or np.any(adapt_image[mask] != 0)
    ):
        raise ValueError("Archive observation or reference has unexpected geometry or values.")
    return MockInputs(
        image, noise, psf, mask, adapt_image, m1["source_pixels"],
        tuple(grid["pixel_center_bbox"]), grid["nsrc"], dpix, nsub,
        stage_from_payload(payload_a), stage_from_payload(payload_m1),
        m1["stage_m1_medians"], manifest["position_likelihood"],
        manifest["injected_mass"], hashes,
    )


def uniform_parameter(name, value, bounds):
    param = ParamU(name, float(value), prior_type="uniform", prior_settings=bounds, limits=bounds)
    param.to_dynamic()
    return param


def _fixed_sie(mock: MockInputs):
    point = mock.m1_point
    return [
        SIE(
            theta_E=point["theta_E"], e1=point["e1_mass"], e2=point["e2_mass"],
            center_x=point["center_x_mass"], center_y=point["center_y_mass"],
        ),
        Shear(gamma1=point["gamma1"], gamma2=point["gamma2"]),
    ]


def _build_model(mock, mass, source, dtype, evidence_logdet, lens_light=(), *, adaptive=True):
    dtype = np.dtype(dtype)
    if dtype == np.dtype(np.float64) and not jax.config.x64_enabled:
        raise ValueError("Enable JAX Float64 before building a Float64 model.")
    return PixelizedImageProbModelOperator(
        image_data=mock.image.astype(dtype),
        noise_map=mock.noise.astype(dtype),
        psf_kernel=mock.psf.astype(dtype),
        dpix=mock.dpix,
        phys_model=PhysicalModel(lens_mass=mass, source_light=[source], lens_light=list(lens_light)),
        mask=mock.mask,
        nsub=mock.nsub,
        position_likelihood=mock.position_likelihood,
        fixed_source_bbox=tuple(dtype.type(value) for value in mock.bbox),
        adapt_image=mock.adapt_image.astype(dtype) if adaptive else None,
        solver_type="pcg",
        block_size=10,
        evidence_logdet=evidence_logdet,
        evidence_block_size=10,
    )


def verify_reference(mock: MockInputs) -> dict:
    """Verify the archived arc against the public component renderer."""
    source = PixelizedSourceModel(
        n=mock.n, log_lambda_reg=mock.m1_point[LOG_LAMBDA],
        regularization_type="first-order", interpolation="bilinear",
    )
    model = _build_model(mock, _fixed_sie(mock), source, np.float32, "block", adaptive=False)
    with ck.ActiveContext(model):
        arc, _, _ = model.reconstruct_component_images(jnp.asarray(mock.source_pixels), None)
    difference = np.asarray(arc) - mock.adapt_image
    np.testing.assert_allclose(arc, mock.adapt_image, rtol=2e-5, atol=2e-5)
    return {
        "producer": "archived SIE plus shear M1",
        **source_interpolation_metadata("bilinear"),
        "model_point": mock.m1_point,
        "max_abs_renderer_difference": float(np.max(np.abs(difference))),
        "positive_active_percentile_99_5": float(np.percentile(
            np.maximum(mock.adapt_image[~mock.mask], 0), ADAPT_REFERENCE_PERCENTILE,
        )),
    }


def build_m2(mock: MockInputs):
    """Calibrate adaptive strengths while fixing the SIE plus shear M1 mass."""
    source = PixelizedSourceModel(
        n=mock.n,
        log_lambda_reg=uniform_parameter(LOG_LAMBDA, mock.m1_point[LOG_LAMBDA], M2_BOUNDS[LOG_LAMBDA]),
        adaptive_reg_rho=uniform_parameter(RHO, 2.0, M2_BOUNDS[RHO]),
        regularization_type="first-order", interpolation=INTERPOLATION,
    )
    return _build_model(mock, _fixed_sie(mock), source, np.float32, "overlap")


def build_m3(
    mock: MockInputs, strengths: dict, *, dtype=np.float32,
    evidence_logdet="overlap", lens_light=(),
):
    """Build the six-parameter EPL plus shear M3 with fixed M2 strengths."""
    stage = mock.stage_m1
    mass = [
        EPL(
            theta_E=stage.gaussian("theta_E", model="EPL", attr="theta_E", limits=(0.0, 5.0)),
            gamma=uniform_parameter("gamma", 2.0, (1.0, 3.0)),
            e1=stage.gaussian("e1_mass", model="EPL", attr="e1", limits=(-1.0, 1.0)),
            e2=stage.gaussian("e2_mass", model="EPL", attr="e2", limits=(-1.0, 1.0)),
            center_x=mock.stage_a.fixed("center_x_mass"),
            center_y=mock.stage_a.fixed("center_y_mass"),
        ),
        Shear(
            gamma1=stage.gaussian("gamma1", model="Shear", attr="gamma1", limits=(-0.5, 0.5)),
            gamma2=stage.gaussian("gamma2", model="Shear", attr="gamma2", limits=(-0.5, 0.5)),
        ),
    ]
    source = PixelizedSourceModel(
        n=mock.n, log_lambda_reg=float(strengths[LOG_LAMBDA]),
        adaptive_reg_rho=float(strengths[RHO]),
        regularization_type="first-order", interpolation=INTERPOLATION,
    )
    return _build_model(mock, mass, source, dtype, evidence_logdet, lens_light)


def _write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def _run_sampler(model, tag, output):
    prior, specs = make_prior_transformation(model)
    names = [spec.name for spec in specs]
    n_live, n_eff = BUDGETS[tag]
    likelihood = make_likelihood(
        model, vectorized=True, dtype=model.image_data.dtype, vectorized_chunk_size=CHUNK,
    )
    sampler = Sampler(
        prior, likelihood, n_dim=len(names), n_live=n_live,
        n_batch=OUTER_BATCH, vectorized=True, seed=SEED,
    )
    print(f"Starting {tag.upper()} with {names}", flush=True)
    start = perf_counter()
    if not sampler.run(n_eff=n_eff, verbose=True):
        raise RuntimeError(f"{tag.upper()} stopped before reaching the Nautilus target.")
    samples, log_weights, scores = sampler.posterior()
    weights = np.exp(log_weights - np.max(log_weights))
    weights /= weights.sum()
    np.savez_compressed(
        output / f"{tag}_overlap.npz",
        samples=samples, weights=weights, log_likelihood=scores, names=names,
    )
    record = {
        "log_z": float(sampler.log_z),
        "likelihood_calls": sampler.n_like,
        "posterior_effective_samples": float(1 / np.sum(weights**2)),
        "sampling_seconds": perf_counter() - start,
        "n_live": n_live,
        "n_eff_target": n_eff,
        "n_batch": OUTER_BATCH,
        "vectorized_chunk_size": CHUNK,
        "seed": SEED,
        "priors": [dict(name=s.name, type=s.prior_type, settings=s.settings, limits=s.limits) for s in specs],
        "solver": model.solver_type,
        "pcg_max_iter": model.pcg_max_iter,
        "pcg_rtol": model.pcg_rtol,
        "block_size": model.block_size,
        "evidence_logdet": model.evidence_logdet,
        "evidence_block_size": model.evidence_block_size,
        "dtype": str(model.image_data.dtype),
    }
    return samples, weights, scores, names, record


def _diagnostics(model, theta):
    result, diag = model.evaluate_with_diagnostics(jnp.asarray(theta, dtype=model.image_data.dtype))
    if not diag.valid or not diag.evidence_valid:
        raise RuntimeError(diag.invalid_reason or diag.evidence_invalid_reason)
    return {
        "score": float(result.log_evidence),
        "primary_valid": bool(diag.valid),
        "evidence_valid": bool(diag.evidence_valid),
        "pcg_iterations": int(result.solver_info.n_iter),
        "chi2": diag.chi2,
        "position_penalty": diag.position.penalty,
        "max_source_separation": diag.position.max_separation,
    }


def run_m2(mock, output):
    model = build_m2(mock)
    samples, weights, scores, names, record = _run_sampler(model, "m2", output)
    medians = [weighted_quantile(samples[:, i], weights, (0.5,))[0] for i in range(len(names))]
    bounds = [M2_BOUNDS[name] for name in names]
    likelihood = make_likelihood(model, dtype=model.image_data.dtype)

    def loss(theta):
        return -likelihood(theta)

    starts = [samples[np.argmax(scores)], np.asarray(medians)]
    candidates = [*starts]
    for start in starts:
        fit = minimize(loss, start, method="Powell", bounds=bounds,
                       options={"xtol": 1e-4, "ftol": 1e-9})
        candidates.append(fit.x)

    def boundary_point(log_lambda, rho):
        parameters = {LOG_LAMBDA: log_lambda, RHO: rho}
        return [parameters[name] for name in names]

    boundary_scores = {}
    for rho in M2_BOUNDS[RHO]:
        fit = minimize_scalar(
            lambda log_lambda: loss(boundary_point(log_lambda, rho)),
            bounds=M2_BOUNDS[LOG_LAMBDA], method="bounded", options={"xatol": 1e-4},
        )
        candidates.append(boundary_point(fit.x, rho))
        boundary_scores[str(rho)] = -float(fit.fun)
    best = min(candidates, key=loss)
    strengths = dict(zip(names, map(float, best)))
    _write_json(output / "m2_calibration.json", {
        **record,
        "selected": strengths,
        "lambda_reg": float(np.exp(strengths[LOG_LAMBDA])),
        "selection": "highest score among posterior peak, medians, Powell refinements and rho bounds",
        "rho_boundary_scores": boundary_scores,
        "quantiles_16_50_84": {
            name: weighted_quantile(samples[:, i], weights, (0.16, 0.5, 0.84)).tolist()
            for i, name in enumerate(names)
        },
        "diagnostics": _diagnostics(model, best),
    })
    print(f"M2 selected {strengths}", flush=True)
    return strengths


def _fit_points(mock, samples, weights, scores, names):
    medians = [weighted_quantile(samples[:, i], weights, (0.5,))[0] for i in range(len(names))]
    return {
        "injected_free_mass": [mock.injected_mass[name] for name in names],
        "posterior_marginal_medians": medians,
        "posterior_peak": samples[np.argmax(scores)],
    }


def run_m3(mock, strengths, output):
    model = build_m3(mock, strengths)
    samples, weights, scores, names, record = _run_sampler(model, "m3", output)
    points = _fit_points(mock, samples, weights, scores, names)
    penalized = ~np.isfinite(scores) | (scores <= -1e10)
    _write_json(output / "m3_fit.json", {
        **record,
        "fixed_strengths": strengths,
        "injected_slope": mock.injected_mass["gamma"],
        "slope_quantiles_2_5_16_50_84_97_5": weighted_quantile(
            samples[:, names.index("gamma")], weights, (0.025, 0.16, 0.5, 0.84, 0.975),
        ).tolist(),
        "points": {
            label: dict(parameters=dict(zip(names, map(float, theta))), **_diagnostics(model, theta))
            for label, theta in points.items()
        },
        "penalized_sampler_records": int(np.count_nonzero(penalized)),
        "posterior_weight_of_penalized_records": float(weights[penalized].sum()),
    })


def _masked_psf_matrix(mask, kernel):
    """Independent sparse circular PSF on the active native pixels."""
    active = np.flatnonzero(~mask)
    height, width = mask.shape
    lookup = np.full(mask.size, -1, dtype=np.int32)
    lookup[active] = np.arange(active.size)
    rows, cols = np.unravel_index(active, mask.shape)
    dy, dx = np.indices(kernel.shape).reshape(2, -1)
    in_rows = (rows[:, None] - dy + kernel.shape[0] // 2) % height
    in_cols = (cols[:, None] - dx + kernel.shape[1] // 2) % width
    source_columns = lookup[in_rows * width + in_cols]
    output_rows = np.broadcast_to(np.arange(active.size)[:, None], source_columns.shape)
    values = np.broadcast_to(kernel.ravel(), source_columns.shape)
    used = source_columns >= 0
    return sparse.csr_matrix(
        (values[used], (output_rows[used], source_columns[used])),
        shape=(active.size, active.size), dtype=np.float64,
    )


def first_order_precision_banded(scale, n, strength, spacing):
    """Construct the complete weighted first-order prior in lower band form."""
    node_scale = np.asarray(scale, dtype=np.float64).reshape(n, n)
    wx = np.sqrt(node_scale[:, :-1] * node_scale[:, 1:])
    wy = np.sqrt(node_scale[:-1, :] * node_scale[1:, :])
    diagonal = np.zeros((n, n))
    diagonal[:, :-1] += wx
    diagonal[:, 1:] += wx
    diagonal[:-1, :] += wy
    diagonal[1:, :] += wy
    diagonal[-1, :] += 1
    diagonal[:, -1] += 1
    nodes = np.arange(n * n).reshape(n, n)
    bands = np.zeros((n + 1, n * n))
    bands[0] = diagonal.ravel()
    bands[1, nodes[:, :-1].ravel()] = -wx.ravel()
    bands[n, nodes[:-1, :].ravel()] = -wy.ravel()
    return (strength / spacing**2) * bands


def complete_logdet_ratio(whitened_design, prior_bands):
    """Evaluate the complete determinant ratio via the data-space identity."""
    chol_prior = linalg.cholesky_banded(prior_bands, lower=True)
    inverse_times_design_t = linalg.cho_solve_banded((chol_prior, True), whitened_design.T)
    covariance = np.eye(whitened_design.shape[0]) + whitened_design @ inverse_times_design_t
    chol = linalg.cholesky(covariance, lower=True)
    return float(2 * np.log(np.diag(chol)).sum())


def _shared_evaluation(model):
    """Extract both production determinant scores from one primary solve."""
    @jax.jit
    def evaluate(theta):
        with ck.ActiveContext(model):
            model.fill_params(theta)
            result, details = model._evaluate(
                compute_evidence=True, include_position=True, return_details=True,
            )
            bbox = result.source_bbox
            stencil = model.sim_obj.precompute_operator_data(*bbox)
            logdet_p_block = model.sim_obj.logdet_block_diag(details.block_chols)
            logdet_h_block = model._regularization_logdet(bbox, details.scale, details.reg_blocks)
            logdet_h_block += model.sim_obj.n_source_pixels * model.source_model.log_lambda_reg.value
            block_ratio = logdet_p_block - logdet_h_block
            chi2 = jnp.sum(((model.data_1d - details.model_source) / model.noise_1d) ** 2)
            source = result.source_pixels
            prior_energy = details.lambda_reg * jnp.dot(
                source, model.reg_builder.matvec_free(source, *bbox, scale=details.scale)
            )
            position_penalty = model._position_likelihood_penalty_jax()
            common = (
                -0.5 * chi2 - 0.5 * prior_energy
                - 0.5 * model.data_1d.size * jnp.log(2.0 * jnp.pi) - 0.5 * model.logdet_C
                + position_penalty
            )
        return result, details.evidence_valid, details.scale, details.lambda_reg, stencil, {
            "common": common, "block_ratio": block_ratio, "chi2": chi2,
            "prior_energy": prior_energy, "position_penalty": position_penalty,
        }

    return evaluate


def _matched_points(mock, output):
    with np.load(output / "m3_overlap.npz") as posterior:
        samples = posterior["samples"]
        names = posterior["names"].tolist()
        weights = posterior["weights"]
        scores = posterior["log_likelihood"]
    order = np.argsort(samples[:, names.index("gamma")])
    # Stratify by posterior mass in slope order, retaining the full mass vector.
    strata = np.searchsorted(np.cumsum(weights[order]), (np.arange(16) + 0.5) / 16)
    points = _fit_points(mock, samples, weights, scores, names)
    points.update({f"slope_stratum_{i + 1:02}": samples[order[j]] for i, j in enumerate(strata)})
    return names, points


def _summarize_comparison(rows):
    exact = np.array([row["full"] for row in rows])
    summary = {"full_best": rows[int(np.argmax(exact))]["point"]}
    for method in ("block", "overlap"):
        scores = np.array([row[method] for row in rows])
        delta = scores - exact
        offset = np.median(delta)
        centered = np.abs(delta - offset)
        summary[method] = {
            "median_signed_offset": float(offset),
            "min_signed_offset": float(np.min(delta)),
            "max_signed_offset": float(np.max(delta)),
            "median_absolute_offset": float(np.median(np.abs(delta))),
            "centered_absolute_error_p95": float(np.percentile(centered, 95)),
            "centered_rmse": float(np.sqrt(np.mean(centered**2))),
            "spearman_rank_correlation": float(stats.spearmanr(exact, scores).statistic),
            "kendall_rank_correlation": float(stats.kendalltau(exact, scores).statistic),
            "best": rows[int(np.argmax(scores))]["point"],
        }
    return summary


def compare_evidence(mock, output, configuration):
    fit_configuration = json.loads((output / "configuration.json").read_text())
    for key in ("inputs_sha256", "m2_m3", "source_node_bbox"):
        if fit_configuration[key] != json.loads(json.dumps(configuration[key])):
            raise ValueError(f"Saved M3 comparison inputs differ in {key}. Refit M2 and M3.")
    m2 = json.loads((output / "m2_calibration.json").read_text())
    m3 = json.loads((output / "m3_fit.json").read_text())
    if m3["fixed_strengths"] != m2["selected"]:
        raise ValueError("Saved M3 strengths differ from the M2 calibration.")
    model = build_m3(mock, m3["fixed_strengths"])
    for key, expected in {
        "solver": model.solver_type,
        "pcg_max_iter": model.pcg_max_iter,
        "pcg_rtol": model.pcg_rtol,
        "block_size": model.block_size,
        "evidence_logdet": model.evidence_logdet,
        "evidence_block_size": model.evidence_block_size,
        "dtype": str(model.image_data.dtype),
    }.items():
        if m3[key] != expected:
            raise ValueError(f"Saved M3 evaluation settings differ in {key}.")
    names, points = _matched_points(mock, output)
    _, specs = make_prior_transformation(model)
    if names != [spec.name for spec in specs]:
        raise ValueError("Saved M3 parameter order differs from the comparison model.")
    noise = mock.noise[~mock.mask].astype(np.float64)
    psf_matrix = _masked_psf_matrix(mock.mask, mock.psf)
    evaluate = _shared_evaluation(model)
    rows = []
    for label, theta in points.items():
        result, valid, scale, strength, stencil, terms = jax.device_get(
            evaluate(jnp.asarray(theta, dtype=model.image_data.dtype))
        )
        if not valid or not result.solver_info.converged:
            raise RuntimeError(f"Shared primary solve or Overlap score failed at {label}.")
        ray_to_native = np.broadcast_to(stencil.agg_segment_ids[:, None], stencil.indices.shape)
        geometric_mapping = sparse.csr_matrix(
            (stencil.weights.astype(np.float64).ravel() / mock.nsub**2,
             (ray_to_native.ravel(), stencil.indices.ravel())),
            shape=(noise.size, mock.n**2),
        )
        design = (psf_matrix @ geometric_mapping).toarray() / noise[:, None]
        predicted = noise * (design @ result.source_pixels)
        model_active = result.model_image[~mock.mask]
        np.testing.assert_allclose(predicted, model_active, rtol=2e-5, atol=2e-5)
        xmin, xmax = map(float, result.source_bbox[:2])
        prior = first_order_precision_banded(scale, mock.n, float(strength),
                                             (xmax - xmin) / (mock.n - 1))
        complete_ratio = complete_logdet_ratio(design, prior)
        common = float(terms["common"])
        block_ratio = float(terms["block_ratio"])
        overlap = float(result.log_evidence)
        row = {
            "point": label,
            "parameters": dict(zip(names, map(float, theta))),
            "primary_valid": bool(result.solver_info.converged),
            "evidence_valid": bool(valid),
            "pcg_iterations": int(result.solver_info.n_iter),
            "common": common,
            "chi2": float(terms["chi2"]),
            "prior_energy": float(terms["prior_energy"]),
            "position_penalty": float(terms["position_penalty"]),
            "block_logdet_ratio": block_ratio,
            "overlap_logdet_ratio": 2 * (common - overlap),
            "full_logdet_ratio": complete_ratio,
            "block": common - 0.5 * block_ratio,
            "overlap": overlap,
            "full": common - 0.5 * complete_ratio,
            "max_abs_independent_prediction_difference": float(np.max(np.abs(predicted - model_active))),
        }
        rows.append(row)
        print(f"Shared-solve comparison {label}: {row['block']:.6f}, "
              f"{overlap:.6f}, {row['full']:.6f}", flush=True)
    _write_json(output / "matched_evidence.json", {
        "fixed_strengths": m2["selected"],
        "comparison_proposal_count": len(rows),
        "full_reference": "Float64 sparse masked design and complete data-space determinant",
        "summary": _summarize_comparison(rows),
        "proposals": rows,
    })


def replay_configuration(mock, reference):
    package = Path(TinyLensGpu.__file__).resolve()
    expected = HERE.parents[2] / "TinyLensGpu" / "__init__.py"
    if package != expected:
        raise RuntimeError(f"Expected checkout import from {expected}, got {package}")
    return {
        "library_revision": subprocess.check_output(
            ["git", "-C", str(expected.parent.parent), "rev-parse", "HEAD"], text=True,
        ).strip(),
        "jax_version": jax.__version__,
        "devices": [device.device_kind for device in jax.devices()],
        "inputs_sha256": mock.hashes,
        "image_shape": mock.image.shape,
        "active_pixels": int(np.count_nonzero(~mock.mask)),
        "dpix_arcsec": mock.dpix,
        "subpixels_per_side": mock.nsub,
        "source_grid_side": mock.n,
        "source_node_bbox": mock.bbox,
        "source_precision": "first-order",
        "reference": reference,
        "m2_m3": {
            **source_interpolation_metadata(INTERPOLATION),
            "projection": "squared per-ray weights with active native image normalization",
            "hermite_coverage_offset": HERMITE_COVERAGE_OFFSET,
            "brightness_percentile": ADAPT_REFERENCE_PERCENTILE,
            "brightness_denominator_floor": ADAPT_REFERENCE_FLOOR,
            "position_likelihood": mock.position_likelihood,
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--phase", choices=("verify", "fit", "compare", "all"), default="all")
    args = parser.parse_args()
    if args.phase != "verify":
        if args.output is None:
            parser.error("--output is required for fits and comparisons")
        if args.output.resolve().is_relative_to(args.archive.resolve()):
            parser.error("--output must be separate from the archive")
    jax.config.update("jax_enable_x64", False)
    jax.config.update("jax_default_matmul_precision", "float32")
    mock = load_mock(args.archive)
    configuration = replay_configuration(mock, verify_reference(mock))
    if args.phase == "verify":
        print(json.dumps(configuration, indent=2))
        return
    args.output.mkdir(parents=True, exist_ok=True)
    if args.phase in ("fit", "all"):
        _write_json(args.output / "configuration.json", configuration)
        strengths = run_m2(mock, args.output)
        jax.clear_caches()
        run_m3(mock, strengths, args.output)
    if args.phase in ("compare", "all"):
        jax.clear_caches()
        compare_evidence(mock, args.output, configuration)


if __name__ == "__main__":
    main()
