"""Benchmark matrix-construction performance for irregular pixelized source models.

This script evaluates the matrix backend performance for:
1. Dense lens mapping matrix construction.
2. Dense blurred lens mapping matrix construction.
3. Dense regularization matrix construction.

It can also report end-to-end reference timings:
1. Solve-only timing.
2. Log-evidence timing.

The script reads observation data from ``data/*.fits`` in the current directory.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

from TinyLensGpu.ForwardSimulation.LensImage.config import SimulatorConfig
from TinyLensGpu.ObservationModel.LensImage import PixelizedImageProbModel
from TinyLensGpu.PhysicalModel.LensImage.Parametric.Mass import SIE
from TinyLensGpu.PhysicalModel.LensImage.Pixelized import (
    IrregularGridConfig,
    MappingConfig,
    PixelizedSourceConfig,
    PixelizedSourceModel,
    RegularizationConfig,
    SolverConfig,
)
from TinyLensGpu.PhysicalModel.LensImage.composite import PhysicalModel
from TinyLensGpu.utils import load_lens_data
from TinyLensGpu.utils.geometry import phi_q2_ellipticity


def _block_until_ready(value: Any) -> Any:
    """Synchronize pending JAX work when available."""
    if hasattr(value, "block_until_ready"):
        value.block_until_ready()
    return value


def _time_callable(fn, n_runs: int) -> Tuple[Dict[str, float], Any]:
    """Time a callable with one warmup run and repeated timed runs."""
    warmup_start = time.perf_counter()
    warmup_value = _block_until_ready(fn())
    warmup_end = time.perf_counter()

    samples: List[float] = []
    for _ in range(int(n_runs)):
        t0 = time.perf_counter()
        _ = _block_until_ready(fn())
        t1 = time.perf_counter()
        samples.append(t1 - t0)

    arr = np.asarray(samples, dtype=np.float64)
    stats = {
        "n_runs": int(n_runs),
        "warmup_s": float(warmup_end - warmup_start),
        "mean_s": float(arr.mean()),
        "std_s": float(arr.std(ddof=1) if int(n_runs) > 1 else 0.0),
        "min_s": float(arr.min()),
        "max_s": float(arr.max()),
    }
    return stats, warmup_value


def _parse_int_list(text: str, *, name: str) -> List[int]:
    """Parse a comma-separated list of positive integers."""
    out: List[int] = []
    for token in text.split(","):
        token = token.strip()
        if not token:
            continue
        value = int(token)
        if value <= 0:
            raise ValueError(f"All values in {name} must be positive, got {value}.")
        out.append(value)
    if not out:
        raise ValueError(f"No valid values parsed for {name}.")
    return out


def _parse_scheme_list(text: str) -> List[str]:
    """Parse and normalize irregular regularization schemes."""
    out: List[str] = []
    for token in text.split(","):
        scheme = token.strip().lower()
        if scheme:
            out.append(scheme)
    if not out:
        raise ValueError("No irregular regularization schemes were provided.")
    return out


def _load_data_from_fits(*, default_dpix: float = 0.074) -> Dict[str, Any]:
    """Load demo data from local FITS files in the ``data`` directory."""
    data_dir = Path(__file__).with_name("data")
    image_path = data_dir / "image.fits"
    noise_path = data_dir / "noise.fits"
    psf_path = data_dir / "psf.fits"
    mask_path = data_dir / "mask.fits"

    required_files = [image_path, noise_path, psf_path, mask_path]
    missing = [str(path) for path in required_files if not path.exists()]
    if missing:
        missing_txt = "\n".join(f"  - {m}" for m in missing)
        raise FileNotFoundError(
            "Required FITS files are missing. Please run sim_data.py in this directory first.\n"
            f"Missing files:\n{missing_txt}"
        )

    image_data, noise_map, psf_kernel, mask = load_lens_data(
        image_path=str(image_path),
        noise_path=str(noise_path),
        psf_path=str(psf_path),
        mask_path=str(mask_path),
    )
    if mask is None:
        raise RuntimeError("mask.fits exists but mask loading returned None.")

    return {
        "noisy_image": image_data,
        "noise_map": noise_map,
        "psf_kernel": psf_kernel,
        "mask": mask,
        "dpix": float(default_dpix),
    }


def _build_prob_model(
    data_dict: Dict[str, Any],
    *,
    n_source_points: int,
    scheme: str,
    mesh_seed: int,
    nonnegative: bool,
) -> PixelizedImageProbModel:
    """Construct a matrix-backend probability model for one irregular-grid case."""
    e1_l, e2_l = phi_q2_ellipticity(90 * np.pi / 180, 0.9)
    sie = SIE(theta_E=1.5, e1=e1_l, e2=e2_l, center_x=0.0, center_y=0.0)

    pix_config = PixelizedSourceConfig(
        grid=IrregularGridConfig(
            n_source_points=int(n_source_points),
            mesh_alpha=1.5,
            mesh_blur_sigma=0.0,
            mesh_method="random",
            mesh_seed=int(mesh_seed),
        ),
        mapping=MappingConfig(
            k_neighbors=5,
            interp_kernel="wendland_c4",
            radius_scale=1.5,
        ),
        regularization=RegularizationConfig(
            scheme=str(scheme),
        ),
        solver=SolverConfig(
            inversion_backend="matrix",
            nonnegative=bool(nonnegative),
        ),
    )
    pix_src_model = PixelizedSourceModel(
        config=pix_config,
        reg_scale=0.05,
        reg_coefficient=1.0,
    )
    phys_model = PhysicalModel(
        lens_mass=[sie],
        source_light=[pix_src_model],
        lens_light=[],
    )
    sim_config = SimulatorConfig(
        dpix=data_dict["dpix"],
        npix=data_dict["noisy_image"].shape[0],
        psf_kernel=data_dict["psf_kernel"],
        mask=data_dict["mask"],
    )
    return PixelizedImageProbModel(
        image_data=data_dict["noisy_image"],
        noise_map=data_dict["noise_map"],
        sim_config=sim_config,
        phys_model=phys_model,
    )


def _solve_once(prob_model: PixelizedImageProbModel):
    """Build an inverter and solve for source coefficients once."""
    data_vector = prob_model.image_data[~prob_model.mask]
    noise_variance = prob_model.noise_map[~prob_model.mask] ** 2
    inverter = prob_model.simulator.build_inverter(
        data_vector=data_vector,
        noise_variance=noise_variance,
        reg_scale=prob_model.pix_src_model.reg_scale.value,
        reg_coefficient=prob_model.pix_src_model.reg_coefficient.value,
    )
    return inverter.solve()


def _log_evidence_once(prob_model: PixelizedImageProbModel):
    """Build an inverter and evaluate log-evidence once."""
    data_vector = prob_model.image_data[~prob_model.mask]
    noise_variance = prob_model.noise_map[~prob_model.mask] ** 2
    inverter = prob_model.simulator.build_inverter(
        data_vector=data_vector,
        noise_variance=noise_variance,
        reg_scale=prob_model.pix_src_model.reg_scale.value,
        reg_coefficient=prob_model.pix_src_model.reg_coefficient.value,
    )
    return inverter.log_evidence()


def _best_by_metric(cases: List[Dict[str, Any]], metric_name: str) -> Optional[Dict[str, Any]]:
    """Return the best (lowest-mean-time) case for one timing metric."""
    eligible = [case for case in cases if metric_name in case["timing"]]
    if not eligible:
        return None
    best = min(eligible, key=lambda c: c["timing"][metric_name]["mean_s"])
    return {
        "metric": metric_name,
        "mean_s": float(best["timing"][metric_name]["mean_s"]),
        "n_source_points": int(best["n_source_points"]),
        "scheme": str(best["scheme"]),
        "n_source": int(best["n_source"]),
        "n_data": int(best["n_data"]),
    }


def _iter_cases(
    n_source_points: Iterable[int],
    schemes: Iterable[str],
 ) -> Iterable[Tuple[int, str]]:
    """Generate irregular-grid benchmark cases."""
    for n_src in n_source_points:
        for scheme in schemes:
            yield int(n_src), str(scheme)


def _run_benchmark(
    *,
    n_runs: int,
    n_source_points: List[int],
    schemes: List[str],
    mesh_seed: int,
    blur_method: str,
    include_solve: bool,
    include_log_evidence: bool,
    nonnegative: bool,
) -> Dict[str, Any]:
    """Run all irregular-grid matrix-construction benchmark cases."""
    data_dict = _load_data_from_fits(default_dpix=0.074)
    cases: List[Dict[str, Any]] = []

    for n_src, scheme in _iter_cases(n_source_points, schemes):
        print(f"[Case] n_source_points={n_src}, scheme={scheme}, backend=matrix")
        prob_model = _build_prob_model(
            data_dict=data_dict,
            n_source_points=n_src,
            scheme=scheme,
            mesh_seed=mesh_seed,
            nonnegative=nonnegative,
        )

        reg_scale = prob_model.pix_src_model.reg_scale.value
        reg_coeff = prob_model.pix_src_model.reg_coefficient.value

        timing: Dict[str, Dict[str, float]] = {}

        mapping_stats, mapping_warmup = _time_callable(
            lambda: prob_model.simulator.build_lens_mapping_matrix(),
            n_runs=n_runs,
        )
        blurred_mapping_stats, blurred_mapping_warmup = _time_callable(
            lambda: prob_model.simulator.build_blurred_lens_mapping_matrix(method=blur_method),
            n_runs=n_runs,
        )
        reg_stats, reg_warmup = _time_callable(
            lambda: prob_model.simulator.build_regularization_matrix(reg_scale=reg_scale, reg_coefficient=reg_coeff),
            n_runs=n_runs,
        )
        timing["mapping_matrix"] = mapping_stats
        timing["blurred_mapping_matrix"] = blurred_mapping_stats
        timing["regularization_matrix"] = reg_stats

        if include_solve:
            solve_stats, _ = _time_callable(lambda: _solve_once(prob_model), n_runs=n_runs)
            timing["solve_only"] = solve_stats

        if include_log_evidence:
            logev_stats, _ = _time_callable(lambda: _log_evidence_once(prob_model), n_runs=n_runs)
            timing["log_evidence_total"] = logev_stats

        n_data, n_source = np.asarray(mapping_warmup).shape
        blurred_n_data, blurred_n_source = np.asarray(blurred_mapping_warmup).shape
        reg_shape = np.asarray(reg_warmup).shape

        cases.append(
            {
                "backend": "matrix",
                "n_source_points": int(n_src),
                "scheme": str(scheme),
                "nonnegative": bool(nonnegative),
                "n_source": int(n_source),
                "n_data": int(n_data),
                "mapping_matrix_shape": [int(n_data), int(n_source)],
                "blurred_mapping_matrix_shape": [int(blurred_n_data), int(blurred_n_source)],
                "regularization_matrix_shape": [int(reg_shape[0]), int(reg_shape[1])],
                "timing": timing,
            }
        )

    metric_order = [
        "mapping_matrix",
        "blurred_mapping_matrix",
        "regularization_matrix",
        "solve_only",
        "log_evidence_total",
    ]
    best_cases = {}
    for metric in metric_order:
        best = _best_by_metric(cases, metric)
        if best is not None:
            best_cases[metric] = best

    return {
        "benchmark_name": "irr_matrix_construction",
        "backend": "matrix",
        "data_source": "fits",
        "n_runs": int(n_runs),
        "mesh_seed": int(mesh_seed),
        "blur_method": str(blur_method),
        "include_solve": bool(include_solve),
        "include_log_evidence": bool(include_log_evidence),
        "nonnegative": bool(nonnegative),
        "n_source_points": [int(v) for v in n_source_points],
        "schemes": [str(v) for v in schemes],
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "cases": cases,
        "best_cases": best_cases,
    }


def build_cli_parser() -> argparse.ArgumentParser:
    """Build command-line interface for benchmark configuration."""
    parser = argparse.ArgumentParser(
        description="Benchmark mapping/regularization matrix construction speed (irregular grid, matrix backend)."
    )
    parser.add_argument("--n-runs", type=int, default=5)
    parser.add_argument("--n-source-points", type=str, default="800,1500,2500")
    parser.add_argument(
        "--schemes",
        type=str,
        default="irregular_gp_exp,irregular_gp_matern32",
    )
    parser.add_argument("--mesh-seed", type=int, default=42)
    parser.add_argument("--blur-method", choices=["fft", "matmul"], default="fft")
    parser.add_argument("--nonnegative", action="store_true")
    parser.add_argument("--include-solve", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include-log-evidence", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("bench_matrix_construction_results.json"),
    )
    return parser


def main() -> None:
    """Parse arguments, run benchmark, and save results."""
    parser = build_cli_parser()
    args = parser.parse_args()

    if int(args.n_runs) <= 0:
        raise ValueError("--n-runs must be positive.")

    n_source_points = _parse_int_list(args.n_source_points, name="n_source_points")
    schemes = _parse_scheme_list(args.schemes)

    payload = _run_benchmark(
        n_runs=int(args.n_runs),
        n_source_points=n_source_points,
        schemes=schemes,
        mesh_seed=int(args.mesh_seed),
        blur_method=str(args.blur_method),
        include_solve=bool(args.include_solve),
        include_log_evidence=bool(args.include_log_evidence),
        nonnegative=bool(args.nonnegative),
    )

    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
