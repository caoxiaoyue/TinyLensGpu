"""Is the M5 likelihood gain a physical critical-curve effect or a numerical one?

Evaluate the current Vegetti-style PJ model, with the intentional tidal
approximation ``rt = sqrt(b_subhalo * theta_E_host)``. The historical SIS
likelihood gain that motivated this probe is not a prediction for PJ; SIS
deflection is constant in magnitude, not the point-mass ``theta_E**2 / r`` law.
Use current PJ caches and model.py globals matching the fitted comparison.

This probe reports, with the subhalo on and off at the same smooth parameters:

  * the source-plane displacement |delta beta| at the marked image positions,
  * the two Jacobian eigenvalues and mu at those positions,
  * the change in ln|mu| across the image plane and where it is largest,
  * an lnL scan in b_subhalo and in subhalo position.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import jax.numpy as jnp

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
parser.add_argument("--scan", action="store_true", help="also scan lnL")
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

stage_m5 = model._stage_from_payload(model._load_stage("m5"))
median_m5 = {name: float(stage_m5.median(name)) for name in stage_m5.param_names}
image_positions = np.asarray(position_likelihood["positions"], dtype=float)
print(f"marked image positions: {np.round(image_positions, 4).tolist()}")
print(f"M5 PJ subhalo median: b_sub={median_m5['b_subhalo']:.5f}, "
      f"centre=({median_m5['center_x_subhalo']:+.4f}, {median_m5['center_y_subhalo']:+.4f})")


# --------------------------------------------------------------------------- #
# Jacobian / magnification at a set of points
# --------------------------------------------------------------------------- #
def mass_model(b_sub, x_sub, y_sub, smooth):
    """Mass-only current PJ model, static at the supplied point (including zero b)."""
    epl, shear = model._epl_mass_from_medians(smooth)
    subhalo = model._pseudo_jaffe_subhalo(epl.theta_E, image_data.shape)
    subhalo.b_sub.to_static(float(b_sub))
    subhalo.center_x.to_static(float(x_sub))
    subhalo.center_y.to_static(float(y_sub))
    return model.PhysicalModel(
        lens_mass=[epl, shear, subhalo], source_light=[], lens_light=[],
    )


def beta_xy(phys, x, y):
    # keyword arguments: with positional args caskade treats the last one as the
    # flattened parameter array (caskade.decorators.forward).
    bx, by = phys.deflection(x=jnp.asarray(x), y=jnp.asarray(y))
    return np.asarray(bx), np.asarray(by)


def jacobian_properties(phys, x, y, eps=2.0e-4):
    """Return lambda_t, lambda_r, mu from finite differences of beta(x, y)."""
    x = np.atleast_1d(np.asarray(x, dtype=float))
    y = np.atleast_1d(np.asarray(y, dtype=float))
    bx_xp, by_xp = beta_xy(phys, x + eps, y)
    bx_xm, by_xm = beta_xy(phys, x - eps, y)
    bx_yp, by_yp = beta_xy(phys, x, y + eps)
    bx_ym, by_ym = beta_xy(phys, x, y - eps)
    a11 = (bx_xp - bx_xm) / (2 * eps)
    a21 = (by_xp - by_xm) / (2 * eps)
    a12 = (bx_yp - bx_ym) / (2 * eps)
    a22 = (by_yp - by_ym) / (2 * eps)
    trace = a11 + a22
    det = a11 * a22 - a12 * a21
    disc = np.sqrt(np.maximum(trace ** 2 - 4 * det, 0.0))
    lam_p = 0.5 * (trace + disc)
    lam_m = 0.5 * (trace - disc)
    return lam_p, lam_m, 1.0 / det


smooth_med = {k: median_m5[k] for k in
              ("theta_E", "gamma", "e1_mass", "e2_mass",
               "center_x_mass", "center_y_mass", "gamma1", "gamma2")}
phys_off = mass_model(0.0, 0.0, 0.0, smooth_med)
phys_on = mass_model(median_m5["b_subhalo"],
                     median_m5["center_x_subhalo"],
                     median_m5["center_y_subhalo"], smooth_med)

print()
print("=" * 78)
print("perturbation at the marked image positions")
print("=" * 78)
bx_off, by_off = beta_xy(phys_off, image_positions[:, 0], image_positions[:, 1])
bx_on, by_on = beta_xy(phys_on, image_positions[:, 0], image_positions[:, 1])
lam_p_off, lam_m_off, mu_off = jacobian_properties(
    phys_off, image_positions[:, 0], image_positions[:, 1])
lam_p_on, lam_m_on, mu_on = jacobian_properties(
    phys_on, image_positions[:, 0], image_positions[:, 1])
for i in range(len(image_positions)):
    dx = bx_on[i] - bx_off[i]
    dy = by_on[i] - by_off[i]
    print(f"  image {i + 1} at ({image_positions[i, 0]:+.4f}, {image_positions[i, 1]:+.4f})")
    print(f"      |delta beta|        = {np.hypot(dx, dy):.3e} arcsec   "
          f"(pixel scale {model.DPIX})")
    print(f"      lambda_t  off/on    = {lam_m_off[i]:+.5f} / {lam_m_on[i]:+.5f}   "
          f"delta = {lam_m_on[i] - lam_m_off[i]:+.5f}")
    print(f"      lambda_r  off/on    = {lam_p_off[i]:+.5f} / {lam_p_on[i]:+.5f}")
    print(f"      mu        off/on    = {mu_off[i]:+.3f} / {mu_on[i]:+.3f}   "
          f"relative change = {(mu_on[i] - mu_off[i]) / mu_off[i]:+.2%}")

print()
print("=" * 78)
print("change in ln|mu| across the unmasked field")
print("=" * 78)
ny, nx = image_data.shape
grid_y, grid_x = np.mgrid[0:ny, 0:nx]
gx = (grid_x - (nx - 1) / 2.0) * model.DPIX
gy = -((grid_y - (ny - 1) / 2.0) * model.DPIX)
valid = ~np.asarray(feature_mask | circular_mask)
sub = np.zeros_like(gx, dtype=bool)
sub[::2, ::2] = True          # coarse sub-sampling of the light curve
sample = valid & sub
xs, ys = gx[sample], gy[sample]
_, _, mu_o = jacobian_properties(phys_off, xs, ys, eps=5.0e-4)
_, _, mu_n = jacobian_properties(phys_on, xs, ys, eps=5.0e-4)
finite = np.isfinite(mu_o) & np.isfinite(mu_n) & (np.abs(mu_o) > 1e-3)
dln = np.log(np.abs(mu_n[finite])) - np.log(np.abs(mu_o[finite]))
print(f"  sampled {finite.sum()} unmasked pixels")
print(f"  median |delta ln mu| = {np.median(np.abs(dln)):.3e}")
print(f"  mean   |delta ln mu| = {np.mean(np.abs(dln)):.3e}")
print(f"  max    |delta ln mu| = {np.max(np.abs(dln)):.3e}  "
      f"(i.e. {np.exp(np.max(np.abs(dln))):.2f}x magnification change)")
order = np.argsort(-np.abs(dln))[:5]
print("  largest changes at:")
for idx in order:
    print(f"      ({xs[finite][idx]:+.3f}, {ys[finite][idx]:+.3f})  "
          f"delta ln mu = {dln[idx]:+.4f}   mu {mu_o[finite][idx]:+.2f} -> "
          f"{mu_n[finite][idx]:+.2f}")

if not args.scan:
    sys.exit(0)

print()
print("=" * 78)
print("lnL scan")
print("=" * 78)
lkl_m5 = model.build_comparison_likelihood(
    lens_subtracted, noise_map, psf_kernel, feature_mask,
    stage_m2, stage_m3, position_likelihood, s1_package,
    circular_mask=circular_mask, with_subhalo=True,
)
names_m5 = [s.name for s in extract_prior_specs(lkl_m5)]
loglike_m5 = make_likelihood(lkl_m5, vectorized=True)


def evaluate(label, **overrides):
    table = dict(median_m5)
    table.update(overrides)
    value = float(np.asarray(
        loglike_m5(np.asarray([table[n] for n in names_m5], dtype=np.float64))
    ).reshape(-1)[0])
    print(f"  {label:56s} lnL = {value:+12.4f}")
    return value


x0, y0 = median_m5["center_x_subhalo"], median_m5["center_y_subhalo"]
base = evaluate("b_sub = 0 (M4 at M5's shared parameters)", b_subhalo=0.0)
for strength in np.linspace(*model.SUBHALO_B_RANGE, 11):
    evaluate(f"b_sub = {strength:.4f} at the M5 median position",
             b_subhalo=float(strength))
print()
evaluate("subhalo moved 1 pixel in x", center_x_subhalo=x0 + 0.05)
evaluate("subhalo moved 4 pixels in x", center_x_subhalo=x0 + 0.2)
evaluate("subhalo moved 1 pixel in y", center_y_subhalo=y0 + 0.05)
evaluate("subhalo moved 4 pixels in y", center_y_subhalo=y0 + 0.2)
evaluate("subhalo moved 20 pixels in x", center_x_subhalo=x0 + 1.0)
evaluate("subhalo parked in an empty corner (4, 4)",
         center_x_subhalo=4.0, center_y_subhalo=4.0)
