"""Chase the smooth/subhalo ridge to see whether the M5 optimum converges.

Historical SIS runs motivated this ridge search. This version evaluates the
current PJ pipeline, requiring current PJ caches and matching model.py globals;
it does not reproduce the old SIS likelihood or evidence values.

This probe does a coordinate ascent on the M5 likelihood: repeatedly maximise
one parameter by a 1D scan with all the others held at the current best point,
cycling theta_E, gamma and b_subhalo. Two outcomes matter:

  * the sequence of slice maxima converges -> the joint optimum is interior and
    finite; a shared prior that brackets it gives a converged Bayes factor;
  * the sequence marches monotonically (theta_E and gamma falling further at
    every cycle, lnL still rising) -> the ridge is open, the evidence
    difference is set by the prior width, and the Bayes factor is not a
    detection statistic.
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
parser.add_argument("--cycles", type=int, default=4)
parser.add_argument("--half-width", type=float, default=0.020,
                    help="half-width of each 1D scan, in parameter units")
parser.add_argument("--n-scan", type=int, default=21)
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
sigma_theta = stage_m3.std("theta_E")
theta_center = stage_m3.median("theta_E")

lkl_m5 = model.build_comparison_likelihood(
    lens_subtracted, noise_map, psf_kernel, feature_mask,
    stage_m2, stage_m3, position_likelihood, s1_package,
    circular_mask=circular_mask, with_subhalo=True,
)
names = [s.name for s in extract_prior_specs(lkl_m5)]
loglike = make_likelihood(lkl_m5, vectorized=True)

stage_m5 = model._stage_from_payload(model._load_stage("m5"))
point = {n: float(stage_m5.median(n)) for n in stage_m5.param_names}


def evaluate(**overrides):
    table = dict(point)
    table.update(overrides)
    return float(np.asarray(
        loglike(np.asarray([table[n] for n in names], dtype=np.float64))
    ).reshape(-1)[0])


def scan(name, half_width, n):
    """1D scan of ``name`` around its current value; return the best point."""
    centre = point[name]
    lo, hi = centre - half_width, centre + half_width
    if name == "b_subhalo":
        lo, hi = max(lo, 0.0), max(hi, 0.0)
    values = np.linspace(lo, hi, n)
    table = []
    for v in values:
        table.append(evaluate(**{name: float(v)}))
    table = np.asarray(table)
    best = int(np.argmax(table))
    print(f"    {name:18s} scan [{values[0]:+.5f}, {values[-1]:+.5f}]  "
          f"peak at {values[best]:+.5f}  lnL = {table[best]:+.3f}"
          + (f"  (offset {(values[best] - theta_center) / sigma_theta:+.2f} sigma_M3)"
             if name == "theta_E" else ""))
    return float(values[best]), float(table[best])


print()
print(f"start point: theta_E = {point['theta_E']:+.5f} "
      f"({(point['theta_E'] - theta_center) / sigma_theta:+.2f} sigma_M3), "
      f"gamma = {point['gamma']:+.5f}, b_sub = {point['b_subhalo']:+.5f}")
print(f"lnL at the start point = {evaluate():+.3f}")

history = [(float(point["theta_E"]), float(point["gamma"]), evaluate())]
planes = (
    ("theta_E", args.half_width),
    ("gamma", 0.05),
    ("b_subhalo", 0.010),
)
for cycle in range(1, args.cycles + 1):
    print(f"\n  cycle {cycle}")
    for name, half in planes:
        best_value, best_lnl = scan(name, half, args.n_scan)
        point[name] = best_value
    lnl = evaluate()
    history.append((float(point["theta_E"]), float(point["gamma"]), lnl))
    print(f"    -> theta_E = {point['theta_E']:+.5f} "
          f"({(point['theta_E'] - theta_center) / sigma_theta:+.2f} sigma_M3), "
          f"gamma = {point['gamma']:+.5f}, b_sub = {point['b_subhalo']:+.5f}, "
          f"lnL = {lnl:+.3f}")

print()
print("=" * 78)
print("ridge trace")
print("=" * 78)
print(f"  {'cycle':>5s} {'theta_E':>10s} {'offset/sigma':>13s} {'gamma':>10s} {'lnL':>12s}")
for i, (te, g, lnl) in enumerate(history):
    print(f"  {i:5d} {te:10.5f} {(te - theta_center) / sigma_theta:+13.2f} "
          f"{g:10.5f} {lnl:+12.3f}")

te_seq = np.asarray([h[0] for h in history])
print()
if np.all(np.diff(te_seq) < -1.0e-4):
    print("  VERDICT: theta_E falls at every cycle while lnL rises -- this chase has")
    print("  not converged; extend the cycles or widen the scan half-width.")
else:
    print("  VERDICT: the trace flattens, so this is a LOCAL optimum of the")
    print(f"  perturbed likelihood at theta_E = {te_seq[-1]:.5f} "
          f"({(te_seq[-1] - theta_center) / sigma_theta:+.2f} sigma_M3), "
          f"lnL = {history[-1][2]:+.3f}.")
    print("  NOTE: a local optimum is not the end of the story. Re-seed this chase")
    print("  from a run with a wider shared prior: if the new seed converges to a")
    print("  HIGHER lnL, the likelihood holds a ladder of ever-better optima along")
    print("  the subhalo/main-lens degeneracy, and the evidence difference will keep")
    print("  growing with the prior width instead of converging. Compare the")
    print("  converged lnL of each seed before trusting any delta ln Z.")
