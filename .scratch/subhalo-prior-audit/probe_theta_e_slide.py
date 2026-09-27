"""Does the M5 likelihood keep improving below the shared prior edge?

Historical SIS runs motivated this probe through prior-edge accumulation.
This version evaluates the current PJ pipeline and requires current PJ caches
with matching model.py globals; old SIS likelihood values are not reproduced.

EITHER the M5 likelihood has a genuine peak just below the edge, in which case a
wider shared prior brackets it and the Bayes factor converges, OR the likelihood
keeps rising without bound along the smooth/subhalo degeneracy, in which case the
evidence difference is set by the arbitrary prior width and no width makes it a
robust subhalo detection. This probe distinguishes the two by scanning both
likelihoods outside the prior support (the likelihood is not clipped to the
prior limits).
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

print("loading data + cached stages ...")
image_data, noise_map, psf_kernel, _ = load_lens_data(
    image_path=str(model.DATA_DIR / "image.fits"),
    noise_path=str(model.DATA_DIR / "noise.fits"),
    psf_path=str(model.DATA_DIR / "psf.fits"),
)
payloads = {tag: model._load_stage(tag) for tag in ("a", "l", "m1", "m2", "m3")}
medians_a = payloads["a"]["extra"]["medians"]
lens_light_model = payloads["l"]["extra"]["lens_light_model"]
stage_m2 = model._stage_from_payload(payloads["m2"])
stage_m3 = model._stage_from_payload(payloads["m3"])
s1_package = model._validate_source_package(payloads["m1"]["extra"]["s1"])
position_likelihood = model._position_likelihood_from_stage_a(medians_a)
circular_mask = model._make_circular_mask(
    image_data.shape, model.DPIX, radius_arcsec=model.MASK_RADIUS,
)
feature_mask = model.run_stage_b(
    image_data, noise_map, lens_light_model, circular_mask=circular_mask,
)
lens_subtracted = image_data - lens_light_model
prior_lo, prior_hi = model._smooth_mass_prior_bounds(stage_m3, "theta_E")
print(f"current shared theta_E prior = [{prior_lo:.6f}, {prior_hi:.6f}]")
print(f"stage-M3 sigma(theta_E) = {stage_m3.std('theta_E'):.6f}")

stage_m4 = model._stage_from_payload(model._load_stage("m4"))
stage_m5 = model._stage_from_payload(model._load_stage("m5"))
med4 = {n: float(stage_m4.median(n)) for n in stage_m4.param_names}
med5 = {n: float(stage_m5.median(n)) for n in stage_m5.param_names}

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


def evaluate(loglike, names, base, **overrides):
    table = dict(base)
    table.update(overrides)
    return float(np.asarray(
        loglike(np.asarray([table[n] for n in names], dtype=np.float64))
    ).reshape(-1)[0])


sigma_theta = stage_m3.std("theta_E")
theta_m3 = stage_m3.median("theta_E")

print()
print("=" * 78)
print("theta_E slice (all other parameters at each arm's own posterior median)")
print("=" * 78)
print(f"  {'theta_E':>10s} {'offset/sigma':>13s}  {'lnL(M4 null)':>14s} "
      f"{'lnL(M5 +sub)':>14s}  {'M5 - M4':>10s}")
rows = []
for te in np.arange(theta_m3 - 0.020, theta_m3 + 0.0101, 0.001):
    l4 = evaluate(loglike_m4, names_m4, med4, theta_E=float(te))
    l5 = evaluate(loglike_m5, names_m5, med5, theta_E=float(te))
    rows.append((te, l4, l5))
    flag = "  <- prior lower edge" if abs(te - prior_lo) < 5e-4 else ""
    print(f"  {te:10.5f} {(te - theta_m3) / sigma_theta:+13.2f}  {l4:+14.4f} "
          f"{l5:+14.4f}  {l5 - l4:+10.4f}{flag}")

rows = np.asarray(rows)
best_m4 = rows[np.argmax(rows[:, 1])]
best_m5 = rows[np.argmax(rows[:, 2])]
print()
print(f"  M4 likelihood peaks at theta_E = {best_m4[0]:.5f} "
      f"({(best_m4[0] - theta_m3) / sigma_theta:+.2f} sigma_M3), "
      f"lnL = {best_m4[1]:+.3f}")
print(f"  M5 likelihood peaks at theta_E = {best_m5[0]:.5f} "
      f"({(best_m5[0] - theta_m3) / sigma_theta:+.2f} sigma_M3), "
      f"lnL = {best_m5[2]:+.3f}")

print()
print("=" * 78)
print("gamma slice")
print("=" * 78)
sigma_g = stage_m3.std("gamma")
gamma_m3 = stage_m3.median("gamma")
print(f"  {'gamma':>10s} {'offset/sigma':>13s}  {'lnL(M4 null)':>14s} "
      f"{'lnL(M5 +sub)':>14s}")
grows = []
for g in np.arange(gamma_m3 - 0.10, gamma_m3 + 0.0301, 0.01):
    l4 = evaluate(loglike_m4, names_m4, med4, gamma=float(g))
    l5 = evaluate(loglike_m5, names_m5, med5, gamma=float(g))
    grows.append((g, l4, l5))
    print(f"  {g:10.5f} {(g - gamma_m3) / sigma_g:+13.2f}  {l4:+14.4f} {l5:+14.4f}")
grows = np.asarray(grows)
print()
print(f"  M4 likelihood peaks at gamma = {grows[np.argmax(grows[:, 1]), 0]:.5f}")
print(f"  M5 likelihood peaks at gamma = {grows[np.argmax(grows[:, 2]), 0]:.5f}")
print()
print("  Interpretation: if the M5 slice rises monotonically as theta_E decreases")
print("  (or gamma decreases) far outside the prior support, the evidence")
print("  difference is set by the prior width, not by the subhalo.")
