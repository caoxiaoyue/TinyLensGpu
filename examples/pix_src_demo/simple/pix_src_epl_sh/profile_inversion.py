"""Profile one pixelized-source log-evidence evaluation by numerical stage.

This script intentionally mirrors the model setup in ``bench_inversion.py`` and
``bench_inversion_matern32.py`` while using ``nsub=2`` to match the fitting
workflow. It writes diagnostic timing summaries only; no optimization is
performed here.
"""

# pyright: reportMissingImports=false

import json
import os
import time
from pathlib import Path

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

os.chdir(Path(__file__).parent)

import caskade as ck
import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsl
import numpy as np

from TinyLensGpu.Inference import ParamU
from TinyLensGpu.Inference.build_prior import make_prior_transformation
from TinyLensGpu.ObservationModel.LensImage import PixelizedImageProbModel
from TinyLensGpu.PhysicalModel import PhysicalModel, EPL, Shear
from TinyLensGpu.PhysicalModel.LensImage.Pixelized.Light import PixelizedSourceModel
from TinyLensGpu.utils import load_lens_data
from TinyLensGpu.utils.lensing.mapping import build_lens_mapping_matrix, build_source_grid


DPIX = 0.05
NSUB = 2
N_WARMUP = 3
N_MEASURE = 50
EPL_TRUE = dict(theta_E=1.0, gamma=2.2, e1=0.1, e2=0.0, center_x=0.0, center_y=0.0)
SHEAR_TRUE = dict(gamma1=0.05, gamma2=0.05)


def build_prob_model(regularization_type):
    """Build the benchmark pixelized model for one regularization type.

    Parameters
    ----------
    regularization_type : str
        Either ``"first-order"`` or ``"matern32"``.

    Returns
    -------
    tuple
        ``(prob_model, pix_src)`` for the requested regularization.
    """
    image_data, noise_map, psf_kernel, mask = load_lens_data(
        image_path="data/image.fits",
        noise_path="data/noise.fits",
        psf_path="data/psf.fits",
        mask_path="data/mask.fits",
    )

    epl = EPL(
        theta_E=ParamU(
            "theta_E",
            EPL_TRUE["theta_E"],
            prior_type="gaussian",
            prior_settings=[EPL_TRUE["theta_E"], 0.1],
            limits=[0.3, 3.0],
        ),
        gamma=ParamU(
            "gamma",
            EPL_TRUE["gamma"],
            prior_type="gaussian",
            prior_settings=[EPL_TRUE["gamma"], 0.1],
            limits=[1.5, 3.0],
        ),
        e1=ParamU(
            "e1",
            EPL_TRUE["e1"],
            prior_type="gaussian",
            prior_settings=[EPL_TRUE["e1"], 0.1],
            limits=[-0.9, 0.9],
        ),
        e2=ParamU(
            "e2",
            EPL_TRUE["e2"],
            prior_type="gaussian",
            prior_settings=[EPL_TRUE["e2"], 0.1],
            limits=[-0.9, 0.9],
        ),
        center_x=ParamU(
            "center_x",
            EPL_TRUE["center_x"],
            prior_type="gaussian",
            prior_settings=[EPL_TRUE["center_x"], 0.05],
            limits=[-0.5, 0.5],
        ),
        center_y=ParamU(
            "center_y",
            EPL_TRUE["center_y"],
            prior_type="gaussian",
            prior_settings=[EPL_TRUE["center_y"], 0.05],
            limits=[-0.5, 0.5],
        ),
    )
    shear = Shear(
        gamma1=ParamU(
            "gamma1",
            SHEAR_TRUE["gamma1"],
            prior_type="gaussian",
            prior_settings=[SHEAR_TRUE["gamma1"], 0.05],
            limits=[-0.5, 0.5],
        ),
        gamma2=ParamU(
            "gamma2",
            SHEAR_TRUE["gamma2"],
            prior_type="gaussian",
            prior_settings=[SHEAR_TRUE["gamma2"], 0.05],
            limits=[-0.5, 0.5],
        ),
    )

    log_lambda_reg = ParamU(
            "log_lambda_reg",
            0.0,
            prior_type="uniform",
            prior_settings=[-6.907755278982137, 6.907755278982137],
            limits=[-13.815510557964274, 13.815510557964274],
    )
    if regularization_type == "matern32":
        pix_src = PixelizedSourceModel(
            nx=40,
            ny=40,
            regularization_type="matern32",
            log_lambda_reg=lambda_reg,
            kernel_scale=ParamU(
                "kernel_scale",
                0.3,
                prior_type="log_uniform",
                prior_settings=[0.01, 2.0],
                limits=[1e-3, 10.0],
            ),
        )
    else:
        pix_src = PixelizedSourceModel(
            nx=40,
            ny=40,
            regularization_type="first-order",
            log_lambda_reg=lambda_reg,
        )

    phys_model = PhysicalModel(lens_mass=[epl, shear], source_light=[pix_src], lens_light=[])
    epl.theta_E.to_dynamic()
    epl.gamma.to_dynamic()
    epl.e1.to_dynamic()
    epl.e2.to_dynamic()
    epl.center_x.to_dynamic()
    epl.center_y.to_dynamic()
    shear.gamma1.to_dynamic()
    shear.gamma2.to_dynamic()
    pix_src.log_lambda_reg.to_dynamic()
    if regularization_type == "matern32":
        assert pix_src.kernel_scale is not None
        pix_src.kernel_scale.to_dynamic()

    prob_model = PixelizedImageProbModel(
        image_data=image_data,
        noise_map=noise_map,
        psf_kernel=psf_kernel,
        dpix=DPIX,
        phys_model=phys_model,
        mask=mask,
        nsub=NSUB,
    )
    return prob_model, pix_src


def sample_theta(prob_model):
    """Draw one deterministic physical sample using the benchmark prior path."""
    prior, prior_specs = make_prior_transformation(prob_model)
    rng = np.random.default_rng(42)
    unit_sample = rng.uniform(0, 1, size=(len(prior_specs),)).astype(np.float32)
    return np.asarray(prior(unit_sample), dtype=np.float32).tolist()


def block_timed(callable_obj):
    """Return elapsed milliseconds after blocking on the callable result."""
    start = time.perf_counter()
    result = callable_obj()
    jax.block_until_ready(result)
    return (time.perf_counter() - start) * 1e3


def summarize(times):
    """Summarize a list of millisecond timings for JSON output."""
    if times is None:
        return None
    values = np.asarray(times, dtype=float)
    return {
        "median_ms": float(np.median(values)),
        "mean_ms": float(np.mean(values)),
        "std_ms": float(np.std(values)),
        "min_ms": float(np.min(values)),
        "max_ms": float(np.max(values)),
    }


def profile_variant(label, output_path):
    """Profile one regularization variant and write its JSON summary."""
    prob_model, pix_src = build_prob_model(label)
    theta = sample_theta(prob_model)

    with ck.ActiveContext(prob_model):
        prob_model.fill_params(theta)

        def deflect_stage():
            beta_x, _ = prob_model.phys_model.deflection(
                x=prob_model.sim_obj.image_x_active_sub,
                y=prob_model.sim_obj.image_y_active_sub,
            )
            return beta_x

        beta_x, beta_y = prob_model.phys_model.deflection(
            x=prob_model.sim_obj.image_x_active_sub,
            y=prob_model.sim_obj.image_y_active_sub,
        )
        source_bbox = prob_model.sim_obj.infer_source_bbox(beta_x, beta_y)
        source_x_axis, source_y_axis, _, _ = build_source_grid(
            prob_model.sim_obj.source_nx,
            prob_model.sim_obj.source_ny,
            *source_bbox,
        )

        def mapping_stage():
            local_bbox = prob_model.sim_obj.infer_source_bbox(beta_x, beta_y)
            local_x_axis, local_y_axis, _, _ = build_source_grid(
                prob_model.sim_obj.source_nx,
                prob_model.sim_obj.source_ny,
                *local_bbox,
            )
            return build_lens_mapping_matrix(beta_x, beta_y, local_x_axis, local_y_axis)

        mapping_matrix_sub = build_lens_mapping_matrix(beta_x, beta_y, source_x_axis, source_y_axis)

        def aggregate_stage():
            return prob_model.sim_obj._aggregate_mapping_to_native(mapping_matrix_sub)

        def design_stage():
            design_matrix, _ = prob_model.sim_obj.design_matrix()
            return design_matrix

        design_matrix, source_bbox = prob_model.sim_obj.design_matrix()
        reg_matrix, _ = prob_model._regularization_matrix(source_bbox)
        lam = jnp.exp(jnp.asarray(pix_src.log_lambda_reg.value))
        weighted_design = design_matrix / prob_model.noise_1d[:, None]
        rhs = weighted_design.T @ (prob_model.data_1d / prob_model.noise_1d)
        curvature = weighted_design.T @ weighted_design + lam * reg_matrix
        chol = jnp.linalg.cholesky(curvature)
        jax.block_until_ready(chol)

        def regmat_stage():
            matrix, _ = prob_model._regularization_matrix(source_bbox)
            return matrix

        def curvature_stage():
            local_weighted = design_matrix / prob_model.noise_1d[:, None]
            return local_weighted.T @ local_weighted + lam * reg_matrix

        def cholesky_stage():
            return jnp.linalg.cholesky(curvature)

        def solve_stage():
            return jsl.cho_solve((chol, True), rhs)

        def evidence_stage():
            return prob_model._log_evidence()

        stage_callables = {
            "deflect": deflect_stage,
            "mapping": mapping_stage,
            "design": design_stage,
            "regmat": regmat_stage,
            "curvature": curvature_stage,
            "cholesky": cholesky_stage,
            "solve": solve_stage,
            "evidence_total": evidence_stage,
        }
        if prob_model.sim_obj.nsub > 1:
            stage_callables["aggregate"] = aggregate_stage

        raw_times = {name: [] for name in stage_callables}
        for name, func in stage_callables.items():
            for _ in range(N_WARMUP):
                block_timed(func)
            for _ in range(N_MEASURE):
                raw_times[name].append(block_timed(func))

    summaries = {name: summarize(times) for name, times in raw_times.items()}
    if "aggregate" not in summaries:
        summaries["aggregate"] = None

    if "aggregate" in raw_times:
        aggregate_times = np.asarray(raw_times["aggregate"])
    else:
        aggregate_times = 0.0
    psf_only_times = np.asarray(raw_times["design"]) - np.asarray(raw_times["mapping"]) - aggregate_times
    summaries["psf_only"] = summarize(psf_only_times)

    ordered_stages = [
        "deflect",
        "mapping",
        "aggregate",
        "design",
        "psf_only",
        "regmat",
        "curvature",
        "cholesky",
        "solve",
        "evidence_total",
    ]
    total_stats = summaries["evidence_total"]
    assert total_stats is not None
    total_ms = total_stats["median_ms"]

    print(f"\n=== {label} regularization (nsub={NSUB}) ===")
    print(f"{'stage':<28s} {'median (ms)':>12s} {'% of total':>12s}")
    for stage in ordered_stages:
        stats = summaries[stage]
        display_name = "evidence_total (umbrella)" if stage == "evidence_total" else stage
        if stats is None:
            print(f"{display_name:<28s} {'n/a':>12s} {'n/a':>12s}")
            continue
        pct = 100.0 * stats["median_ms"] / total_ms if total_ms != 0.0 else np.nan
        if stage == "evidence_total":
            pct = 100.0
        print(f"{display_name:<28s} {stats['median_ms']:>12.3f} {pct:>12.1f}")

    output = {
        "regularization_type": label,
        "N": N_MEASURE,
        "n_warmup": N_WARMUP,
        "nsub": NSUB,
        "device": str(jax.devices()[0]),
        "Ns": int(prob_model.sim_obj.n_source_pixels),
        "Nd_active": int(prob_model.sim_obj.flat_indices.shape[0]),
        "stages": summaries,
    }
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Saved {output_path}")


def main():
    """Run both diagnostic profiling variants."""
    os.makedirs("output", exist_ok=True)
    profile_variant("first-order", "output/profile_first_order.json")
    profile_variant("matern32", "output/profile_matern32.json")


if __name__ == "__main__":
    main()
