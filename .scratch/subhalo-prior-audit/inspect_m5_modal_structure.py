"""Inspect modal structure using the actual saved StagePosterior prior specs.

Historical SIS prior-edge accumulation motivated this audit; no historical
numerical result is assumed here. All parameter names and prior bounds come
from the chosen cache, so stored SIS and current PJ posteriors retain their
own meanings. This script does not rebuild or evaluate a current likelihood.
Reports weighted quantiles, theta_E structure and parameter correlations.
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]

parser = argparse.ArgumentParser()
parser.add_argument("--out-dir", default="output")
args = parser.parse_args()

out_dir = (ROOT / "examples/pix_src_demo_operator/pipe/subhalo_0946" / args.out_dir)
print(f"reading {out_dir}")

with open(out_dir / "stage_m5.pkl", "rb") as fh:
    payload = pickle.load(fh)

samples = np.asarray(payload["samples"], dtype=float)
weights = np.asarray(payload["weights"], dtype=float)
names = list(payload.get("param_names") or payload["extra"]["param_names"])
prior_specs = {s["name"]: s for s in payload["stage_schema"]["prior_specs"]}
log_z = payload.get("log_z")
print(f"samples = {samples.shape}, log_z = {log_z}")


def wquant(values, q):
    order = np.argsort(values)
    v = values[order]
    w = weights[order]
    cw = np.cumsum(w) - 0.5 * w
    cw /= np.sum(w)
    return float(np.interp(q, cw, v))


print()
print("=" * 78)
print("weighted quantiles of the stage-M5 posterior")
print("=" * 78)
print(f"  {'param':18s} {'1%':>10s} {'16%':>10s} {'50%':>10s} {'84%':>10s} "
      f"{'99%':>10s}  {'prior lo':>10s} {'prior hi':>10s}  {'below 2%?':>9s}")
for i, name in enumerate(names):
    col = samples[:, i]
    q = [wquant(col, x) for x in (0.01, 0.16, 0.50, 0.84, 0.99)]
    lo, hi = col.min(), col.max()
    entry = prior_specs[name]
    bounds = entry["settings"] if entry["prior_type"] == "uniform" else entry["limits"]
    plo, phi = bounds if bounds is not None else (float("-inf"), float("inf"))
    cut = plo + 0.02 * (phi - plo)
    frac = float(np.sum(weights[col < cut]) / np.sum(weights))
    print(f"  {name:18s} {q[0]:10.5f} {q[1]:10.5f} {q[2]:10.5f} {q[3]:10.5f} "
          f"{q[4]:10.5f}  {plo:10.5f} {phi:10.5f}  {100 * frac:8.2f}%")

te_idx = names.index("theta_E")
te = samples[:, te_idx]
lo, hi = prior_specs["theta_E"]["settings"]
print()
print("=" * 78)
print("theta_E posterior shape (saved prior)")
print("=" * 78)
bins = np.linspace(lo, hi, 21)
hist, edges = np.histogram(te, bins=bins, weights=weights)
hist = hist / hist.sum()
for h, e0, e1 in zip(hist, edges[:-1], edges[1:]):
    bar = "#" * int(round(h * 120))
    print(f"  [{e0:.5f}, {e1:.5f}]  {100 * h:6.2f}%  {bar}")
print(f"  mass below the 2% edge ({lo + 0.02 * (hi - lo):.5f}) = "
      f"{100 * np.sum(weights[te < lo + 0.02 * (hi - lo)]) / np.sum(weights):.2f}%")

print()
print("=" * 78)
print("correlation of theta_E with the parameters that can absorb its shift")
print("=" * 78)
te_w = te - wquant(te, 0.5)
for i, name in enumerate(names):
    if i == te_idx:
        continue
    col = samples[:, i]
    c = col - wquant(col, 0.5)
    denom = np.sqrt(np.sum(weights * te_w ** 2) * np.sum(weights * c ** 2))
    rho = float(np.sum(weights * te_w * c) / denom) if denom > 0 else float("nan")
    print(f"  corr(theta_E, {name:20s}) = {rho:+.3f}")

print()
print("=" * 78)
print("the co-moving parameters in the lowest theta_E decile")
print("=" * 78)
cut = wquant(te, 0.10)
mask = te <= cut
print(f"  lowest theta_E decile: theta_E <= {cut:.5f} "
      f"({100 * np.sum(weights[mask]) / np.sum(weights):.1f}% of the weight)")
for i, name in enumerate(names):
    if i == te_idx:
        continue
    col = samples[:, i]
    high = np.sum(weights[~mask] * col[~mask]) / np.sum(weights[~mask])
    loww = np.sum(weights[mask] * col[mask]) / np.sum(weights[mask])
    print(f"    {name:20s} lowest-decile mean = {loww:+10.5f}   "
          f"rest = {high:+10.5f}   shift = {loww - high:+10.5f}")
