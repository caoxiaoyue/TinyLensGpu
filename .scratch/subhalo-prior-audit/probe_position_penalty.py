"""Diagnose the position-likelihood penalty inside the M4/M5 comparison arms.

Explicitly enable the optional position penalty in both current PJ comparison
arms. Stage-A positions are solved from the smooth model; this probe reports
how their source-plane separation and penalty respond to a PJ perturbation.
The default pipeline may disable this penalty. Historical SIS sweep values
are not predictions for the current truncated profile.
"""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "examples/pix_src_demo_operator/pipe/subhalo_0946/model.py"

spec = importlib.util.spec_from_file_location("subhalo_model", MODEL_PATH)
model = importlib.util.module_from_spec(spec)
sys.modules["subhalo_model"] = model
spec.loader.exec_module(model)

import jax.numpy as jnp
import caskade as ck

from TinyLensGpu.Inference.build_prior import make_prior_transformation
from TinyLensGpu.Inference.build_likelihood import make_likelihood
from TinyLensGpu.utils import load_lens_data

model.COMPARISON_USE_POSITION_LIKELIHOOD = True

image_data, noise_map, psf_kernel, _ = load_lens_data(
    image_path=str(model.DATA_DIR / "image.fits"),
    noise_path=str(model.DATA_DIR / "noise.fits"),
    psf_path=str(model.DATA_DIR / "psf.fits"),
)
payloads = {tag: model._load_stage(tag) for tag in ("a", "l", "m1", "m2", "m3")}
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

print("marked image positions (solved from the smooth stage-A model):")
for px, py in position_likelihood["positions"]:
    print(f"    ({px:+.4f}, {py:+.4f})")
print(f"threshold = {position_likelihood['threshold_arcsec']} arcsec, "
      f"min_log_like = {position_likelihood['min_log_like']}")


def prepare(with_subhalo):
    lkl = model.build_comparison_likelihood(
        lens_subtracted, noise_map, psf_kernel, feature_mask, stage_m2, stage_m3,
        position_likelihood, s1_package, circular_mask=circular_mask,
        with_subhalo=with_subhalo,
    )
    prior_fn, prior_specs = make_prior_transformation(lkl)
    names = [s.name for s in prior_specs]
    theta = np.asarray(prior_fn(np.full((1, len(names)), 0.5)))[0].copy()
    return lkl, make_likelihood(lkl, vectorized=True), names, theta


def evaluate(lkl, loglike, names, theta, overrides=None):
    point = theta.copy()
    for key, value in (overrides or {}).items():
        point[names.index(key)] = float(value)
    total = float(np.asarray(loglike(point.reshape(1, -1)))[0])
    with ck.ActiveContext(lkl):
        lkl.fill_params(jnp.asarray(point))
        penalty = float(np.asarray(lkl._position_likelihood_penalty_jax()))
        beta_x, beta_y = lkl.phys_model.deflection(x=lkl._pos_px, y=lkl._pos_py)
    bx, by = np.asarray(beta_x), np.asarray(beta_y)
    dx = bx[:, None] - bx[None, :]
    dy = by[:, None] - by[None, :]
    max_sep = float(np.max(np.sqrt(dx * dx + dy * dy)))
    return max_sep, penalty, total


print()
print("=" * 78)
print("A. M4 (smooth null) at the shared prior centre")
print("=" * 78)
lkl_m4, ll_m4, names_m4, theta_m4 = prepare(False)
sep, pen, tot = evaluate(lkl_m4, ll_m4, names_m4, theta_m4)
print(f"  M4 centre   max_sep={sep:.6f}\"  penalty={pen:+.6e}  "
      f"loglike={tot:+.3f}  (loglike-penalty={tot - pen:+.3f})")

print()
print("=" * 78)
print("B. M5 (smooth + PJ) at the shared prior centre, subhalo swept")
print("=" * 78)
lkl_m5, ll_m5, names_m5, theta_m5 = prepare(True)
sweep = [
    (0.0, 0.0, 0.0),
    (0.0, 0.0, 0.01),
    (0.0, 0.0, 0.05),
    (1.0, 0.0, 0.05),
    (-0.7071, 1.0, 0.05),
    (-1.5, 1.0, 0.05),
    (2.0, 2.0, 0.05),
    (4.5, 4.5, 0.10),
]
rows = []
for x, y, b_sub in sweep:
    sep, pen, tot = evaluate(
        lkl_m5, ll_m5, names_m5, theta_m5,
        {"b_subhalo": b_sub, "center_x_subhalo": x, "center_y_subhalo": y},
    )
    rows.append((x, y, b_sub, sep, pen, tot))
    print(f"  b_sub={b_sub:.2f} at ({x:+.2f},{y:+.2f})   "
          f"max_sep={sep:.6f}\"  penalty={pen:+.6e}  "
          f"loglike={tot:+.3f}  (loglike-penalty={tot - pen:+.3f})")

print()
print("=" * 78)
print("C. check the documented penalty law")
print("=" * 78)
thr = float(position_likelihood["threshold_arcsec"])
minl = float(position_likelihood["min_log_like"])
_, _, _, sep, pen, _ = rows[2]
ratio = max(0.0, sep - thr) / thr
predicted = minl * (1.0 - math.exp(-ratio))
print(f"  max_sep = {sep:.6f} arcsec, ratio = {ratio:.6f}")
print(f"  predicted penalty = {predicted:+.6e}")
print(f"  observed  penalty = {pen:+.6e}")
print(f"  absolute difference = {abs(predicted - pen):.3e}")
print()
print("  A subhalo whose presence moves the two marked images apart by more than")
print("  the 0.1 arcsec threshold in the source plane is suppressed exponentially.")
print("  Those marked positions were themselves solved from the smooth stage-A")
print("  model, so the constraint is not a neutral prior on the comparison.")
