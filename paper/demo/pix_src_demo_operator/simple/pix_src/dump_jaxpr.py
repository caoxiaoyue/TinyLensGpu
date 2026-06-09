"""Dump and summarize the JAXPR for one pixelized-source likelihood call."""

# pyright: reportMissingImports=false, reportCallIssue=false

import json
import os
from pathlib import Path

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

os.chdir(Path(__file__).parent)

import jax
import jax.numpy as jnp
import numpy as np

from TinyLensGpu.Inference import ParamU
from TinyLensGpu.Inference.build_likelihood import make_likelihood
from TinyLensGpu.Inference.build_prior import make_prior_transformation
from TinyLensGpu.ObservationModel.LensImage.pixelized_image_model_operator import PixelizedImageProbModelOperator
from TinyLensGpu.PhysicalModel import PhysicalModel, SIE
from TinyLensGpu.PhysicalModel.LensImage.Pixelized.Light import PixelizedSourceModel
from TinyLensGpu.utils import load_lens_data


DPIX = 0.05
NSUB = 2
SIE_TRUE = dict(theta_E=1.0, e1=0.1, e2=0.0, center_x=0.0, center_y=0.0)
OPS_OF_INTEREST = ("dot_general", "cholesky", "triangular_solve", "fft", "gather", "scatter")


def build_prob_model():
    """Build the first-order benchmark pixelized model used for JAXPR tracing."""
    image_data, noise_map, psf_kernel, mask = load_lens_data(
        image_path="data/image.fits",
        noise_path="data/noise.fits",
        psf_path="data/psf.fits",
        mask_path="data/mask.fits",
    )
    sie = SIE(
        theta_E=ParamU(
            "theta_E",
            SIE_TRUE["theta_E"],
            prior_type="gaussian",
            prior_settings=[1.0, 0.1],
            limits=[0.3, 3.0],
        ),
        e1=ParamU("e1", SIE_TRUE["e1"], prior_type="gaussian", prior_settings=[0.1, 0.1], limits=[-0.9, 0.9]),
        e2=ParamU("e2", SIE_TRUE["e2"], prior_type="gaussian", prior_settings=[0.0, 0.1], limits=[-0.9, 0.9]),
        center_x=ParamU("center_x", 0.0, prior_type="gaussian", prior_settings=[0.0, 0.05], limits=[-0.5, 0.5]),
        center_y=ParamU("center_y", 0.0, prior_type="gaussian", prior_settings=[0.0, 0.05], limits=[-0.5, 0.5]),
    )
    pix_src = PixelizedSourceModel(
        nx=40,
        ny=40,
        regularization_type="first-order",
        lambda_reg=ParamU(
            "lambda_reg",
            1.0,
            prior_type="log_uniform",
            prior_settings=[1e-3, 1e3],
            limits=[1e-6, 1e6],
        ),
    )
    phys_model = PhysicalModel(lens_mass=[sie], source_light=[pix_src], lens_light=[])
    sie.theta_E.to_dynamic()
    sie.e1.to_dynamic()
    sie.e2.to_dynamic()
    sie.center_x.to_dynamic()
    sie.center_y.to_dynamic()
    pix_src.lambda_reg.to_dynamic()
    return PixelizedImageProbModelOperator(
        image_data=image_data,
        noise_map=noise_map,
        psf_kernel=psf_kernel,
        dpix=DPIX,
        phys_model=phys_model,
        mask=mask,
        nsub=NSUB,
    )


def sample_theta(prob_model):
    """Draw one deterministic physical sample using the benchmark prior path."""
    prior, prior_specs = make_prior_transformation(prob_model)
    rng = np.random.default_rng(42)
    unit_sample = rng.uniform(0, 1, size=(len(prior_specs),)).astype(np.float32)
    return jnp.asarray(prior(unit_sample))


def walk_jaxpr(jaxpr_like, counts):
    """Recursively count primitive equations in a JAXPR-like object."""
    if hasattr(jaxpr_like, "jaxpr"):
        jaxpr_like = jaxpr_like.jaxpr
    if not hasattr(jaxpr_like, "eqns"):
        return

    for eqn in jaxpr_like.eqns:
        prim_name = eqn.primitive.name
        counts["total_eqns_recursive"] += 1
        counts["primitive_counts"][prim_name] = counts["primitive_counts"].get(prim_name, 0) + 1
        for value in eqn.params.values():
            if hasattr(value, "jaxpr") or hasattr(value, "eqns"):
                walk_jaxpr(value, counts)
            elif isinstance(value, (tuple, list)):
                for item in value:
                    if hasattr(item, "jaxpr") or hasattr(item, "eqns"):
                        walk_jaxpr(item, counts)


def summarize_ops(primitive_counts):
    """Return counts for the requested operations of interest."""
    result = {name: 0 for name in ["dot_general", "cholesky", "triangular_solve", "fft", "gather", "scatter"]}
    for prim_name, count in primitive_counts.items():
        if prim_name in ("add", "add_p", "mul", "mul_p"):
            continue
        if prim_name == "dot_general":
            result["dot_general"] += count
        elif prim_name == "cholesky":
            result["cholesky"] += count
        elif prim_name == "triangular_solve":
            result["triangular_solve"] += count
        elif "fft" in prim_name:
            result["fft"] += count
        elif prim_name == "gather":
            result["gather"] += count
        elif prim_name.startswith("scatter"):
            result["scatter"] += count
    return result


def main():
    """Trace, dump, and summarize the first-order likelihood JAXPR."""
    os.makedirs("output", exist_ok=True)
    prob_model = build_prob_model()
    theta_sample = sample_theta(prob_model)
    loglike_fn = make_likelihood(prob_model, vectorized=False)
    _ = loglike_fn

    try:
        closed_jaxpr = jax.make_jaxpr(prob_model)(theta_sample)
    except Exception:
        def trace_fn(theta):
            return prob_model(theta)

        closed_jaxpr = jax.make_jaxpr(trace_fn)(theta_sample)

    with open("output/jaxpr_loglike.txt", "w") as out_file:
        out_file.write(str(closed_jaxpr))

    counts = {"total_eqns_recursive": 0, "primitive_counts": {}}
    walk_jaxpr(closed_jaxpr, counts)
    op_counts = summarize_ops(counts["primitive_counts"])
    top_level_eqn_count = len(closed_jaxpr.jaxpr.eqns)
    stats = {
        "top_level_eqn_count": top_level_eqn_count,
        "total_eqns_recursive": counts["total_eqns_recursive"],
        "ops_of_interest": op_counts,
        "primitive_counts": dict(sorted(counts["primitive_counts"].items())),
        "device": str(jax.devices()[0]),
        "nsub": NSUB,
    }
    with open("output/jaxpr_stats.json", "w") as out_file:
        json.dump(stats, out_file, indent=2)

    print(f"top_level_eqn_count: {top_level_eqn_count}")
    print("ops_of_interest:")
    for name, count in op_counts.items():
        print(f"  {name}: {count}")
    print("Saved output/jaxpr_loglike.txt")
    print("Saved output/jaxpr_stats.json")


if __name__ == "__main__":
    main()
