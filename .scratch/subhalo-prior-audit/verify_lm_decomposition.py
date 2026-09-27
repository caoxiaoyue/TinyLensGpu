"""Decompose the current PJ M5-M4 likelihood difference at posterior medians.

At fixed posterior-median points, separate the change in the image likelihood
from moving the shared parameters and from switching on the PJ subhalo.
Both terms hold the other parameters fixed; neither is a marginal evidence.

Write L_M4 for the null likelihood (smooth only) and L_M5 for the PJ likelihood.
Setting ``b_subhalo = 0`` turns off its deflection exactly, including the tied
tidal radius ``rt = sqrt(b_subhalo * theta_E_host)``. Then

    delta ln L_smooth = L_M4(theta_smooth = median_M5) - L_M4(theta_smooth = median_M4)
    delta ln L_sub    = L_M5(median_M5) - L_M5(median_M5, b_subhalo = 0)

These are pointwise likelihood diagnostics, not a decomposition of integrated
evidence or a detection verdict. Use current PJ caches with matching model.py
globals; historical SIS outputs are not valid inputs to this current likelihood.

Usage::

    python verify_lm_decomposition.py [--out-dir output]
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "examples/pix_src_demo_operator/pipe/subhalo_0946/model.py"

spec = importlib.util.spec_from_file_location("subhalo_model", MODEL_PATH)
model = importlib.util.module_from_spec(spec)
sys.modules["subhalo_model"] = model
spec.loader.exec_module(model)

from TinyLensGpu.Inference.build_prior import extract_prior_specs
from TinyLensGpu.Inference.build_likelihood import make_likelihood
from TinyLensGpu.utils import load_lens_data

parser = argparse.ArgumentParser()
parser.add_argument("--out-dir", default="output")
args = parser.parse_args()

model.OUT_DIR = MODEL_PATH.parent / args.out_dir

print(f"output dir = {model.OUT_DIR}")
print("loading data + cached stages ...")
image_data, noise_map, psf_kernel, _ = load_lens_data(
    image_path=str(model.DATA_DIR / "image.fits"),
    noise_path=str(model.DATA_DIR / "noise.fits"),
    psf_path=str(model.DATA_DIR / "psf.fits"),
)
payloads = {tag: model._load_stage(tag) for tag in ("a", "l", "m1", "m2", "m3")}
stage_a = model._stage_from_payload(payloads["a"])
stage_m2 = model._stage_from_payload(payloads["m2"])
stage_m3 = model._stage_from_payload(payloads["m3"])
medians_a = payloads["a"]["extra"]["medians"]
lens_light_model = payloads["l"]["extra"]["lens_light_model"]
feature_mask = model.run_stage_b(
    image_data, noise_map, lens_light_model,
    circular_mask=model._make_circular_mask(image_data.shape, model.DPIX,
                                            radius_arcsec=model.MASK_RADIUS),
)
s1_package = model._validate_source_package(payloads["m1"]["extra"]["s1"])
position_likelihood = model._position_likelihood_from_stage_a(medians_a)
circular_mask = model._make_circular_mask(
    image_data.shape, model.DPIX, radius_arcsec=model.MASK_RADIUS,
)
lens_subtracted = image_data - lens_light_model

stage_m4 = model._stage_from_payload(model._load_stage("m4"))
stage_m5 = model._stage_from_payload(model._load_stage("m5"))
median_m4 = {name: float(stage_m4.median(name)) for name in stage_m4.param_names}
median_m5 = {name: float(stage_m5.median(name)) for name in stage_m5.param_names}

lkl_m4 = model.build_comparison_likelihood(
    lens_subtracted, noise_map, psf_kernel, feature_mask,
    stage_m2, stage_m3, position_likelihood, s1_package,
    circular_mask=circular_mask, with_subhalo=False,
)
lkl_m5 = model.build_comparison_likelihood(
    lens_subtracted, noise_map, psf_kernel, feature_mask,
    stage_m2, stage_m3, position_likelihood, s1_package,
    circular_mask=circular_mask, with_subhalo=True,
)
names_m4 = [s.name for s in extract_prior_specs(lkl_m4)]
names_m5 = [s.name for s in extract_prior_specs(lkl_m5)]
loglike_m4 = make_likelihood(lkl_m4, vectorized=True)
loglike_m5 = make_likelihood(lkl_m5, vectorized=True)


def theta(names, values):
    return np.asarray([values[name] for name in names], dtype=np.float64)


def evaluate(loglike, names, values, label, **overrides):
    table = dict(values)
    table.update(overrides)
    value = float(np.asarray(loglike(theta(names, table))).reshape(-1)[0])
    note = ", ".join(f"{k}={v:+.4f}" for k, v in overrides.items())
    print(f"    {label:52s} lnL = {value:+12.4f}"
          + (f"   [{note}]" if note else ""))
    return value


print()
print("=" * 78)
print(f"posterior medians ({args.out_dir})")
print("=" * 78)
for name, value in median_m4.items():
    print(f"    M4 {name:20s} = {value:+.6f}"
          + (f"    M5 = {median_m5[name]:+.6f}"
             f"   delta = {median_m5[name] - value:+.6f}"
             if name in median_m5 else ""))
for name in median_m5:
    if name not in median_m4:
        print(f"    M5 {name:20s} = {median_m5[name]:+.6f}   (subhalo only)")

b_sub = median_m5["b_subhalo"]
radius = float(np.hypot(median_m5["center_x_subhalo"],
                        median_m5["center_y_subhalo"]))
rt = float(np.sqrt(b_sub * median_m5["theta_E"]))
print()
print(f"    PJ b_sub = {b_sub:.5f} arcsec, radius from origin = {radius:.5f} arcsec")
print(f"    rt = sqrt(b_sub * theta_E_host) = {rt:.5f} arcsec "
      "(intentional Einstein-radius approximation)")

print()
print("=" * 78)
print("pointwise likelihood decomposition at the posterior medians")
print("=" * 78)
lnL_m4_m4 = evaluate(loglike_m4, names_m4, median_m4, "L_M4(M4 median)")
lnL_m4_m5 = evaluate(loglike_m4, names_m4, median_m5, "L_M4(M5 smooth median, no subhalo)")
lnL_m5_m5 = evaluate(loglike_m5, names_m5, median_m5, "L_M5(M5 median)")
lnL_m5_off = evaluate(loglike_m5, names_m5, median_m5,
                      "L_M5(M5 median, subhalo switched off)",
                      b_subhalo=0.0)
# M4 has no subhalo coordinates: retain a valid M5 point, replace every shared
# nuisance value with M4's median, and only then switch off the PJ strength.
m4_in_m5 = {**median_m5, **median_m4}
lnL_m5_m4fit = evaluate(loglike_m5, names_m5, m4_in_m5,
                        "L_M5(M4 smooth median, no subhalo)",
                        b_subhalo=0.0)

print()
print("    delta lnL from moving the smooth parameters (M4 likelihood):")
print(f"        L_M4(M5 smooth median) - L_M4(M4 median) = {lnL_m4_m5 - lnL_m4_m4:+.4f}")
print("    delta lnL from switching the subhalo on at fixed smooth parameters:")
print(f"        L_M5(M5 median) - L_M5(M5 median, off)   = {lnL_m5_m5 - lnL_m5_off:+.4f}")
print("    null cross-check at M4's nuisance parameters:")
print(f"        L_M5(M4 median, off) - L_M4(M4 median) = {lnL_m5_m4fit - lnL_m4_m4:+.6f}")
print("    null cross-check at M5's nuisance parameters:")
print(f"        L_M5(M5 median, off) - L_M4(M5 median) = {lnL_m5_off - lnL_m4_m5:+.6f}")
print("    These pointwise differences do not determine the Bayes factor.")
