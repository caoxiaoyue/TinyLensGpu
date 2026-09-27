"""Posterior shape of the localised subhalo search (no JAX needed).

Loads a current PJ stage-M5 (or M4) cache written by ``model.py`` and reports
the weighted subhalo posterior, including cumulative probability near
``b_subhalo = 0``. Historical SIS caches use a different strength parameter
and are not inputs to this current-PJ summary.

Usage::

    python inspect_local_m5.py output/stage_m5.pkl
"""

import pickle
import sys

import numpy as np


def weighted_quantile(values, weights, quantiles):
    order = np.argsort(values)
    v = np.asarray(values)[order]
    w = np.asarray(weights)[order]
    cdf = np.cumsum(w)
    cdf = (cdf - 0.5 * w) / np.sum(w)
    return np.interp(np.asarray(quantiles, dtype=float), cdf, v)


def main(path):
    with open(path, "rb") as handle:
        payload = pickle.load(handle)

    samples = np.asarray(payload["samples"], dtype=float)
    weights = np.asarray(payload["weights"], dtype=float)
    names = list(payload["param_names"])
    extra = payload.get("extra", {})

    print("=" * 72)
    print(f"stage posterior summary: {path}")
    print(f"  log Z = {payload.get('log_z'):+.3f}   samples = {len(samples)}")
    print(f"  pipeline schema = {extra.get('pipeline_schema')}")
    print(f"  role = {extra.get('comparison_role')}")
    print("=" * 72)

    total = weights.sum()
    subhalo_cols = [n for n in names if n.endswith("_subhalo")]
    if not subhalo_cols:
        print("no subhalo parameters in this stage.")
        return

    print("subhalo posterior (median, 68% interval, weighted):")
    for name in subhalo_cols:
        col = samples[:, names.index(name)]
        q16, q50, q84 = weighted_quantile(col, weights, [0.16, 0.50, 0.84])
        print(f"  {name:<20} = {q50:+.4f}  ({q16:+.4f}, {q84:+.4f})")

    strength = samples[:, names.index("b_subhalo")]
    xs = samples[:, names.index("center_x_subhalo")]
    ys = samples[:, names.index("center_y_subhalo")]

    print()
    print("PJ b_sub cumulative weight (is the posterior piling up at 0?):")
    for threshold in (0.001, 0.005, 0.010, 0.0122, 0.020, 0.030, 0.050, 0.072):
        below = weights[strength < threshold].sum() / total
        print(f"  P(b_sub < {threshold:.4f}) = {below:.4f}")

    print()
    print("position (arcsec), weighted posterior:")
    for name, col in (("x_subhalo", xs), ("y_subhalo", ys)):
        q16, q50, q84 = weighted_quantile(col, weights, [0.16, 0.50, 0.84])
        print(f"  {name:<12} = {q50:+.4f}  ({q16:+.4f}, {q84:+.4f})")
    radius = np.hypot(xs, ys)
    q16, q50, q84 = weighted_quantile(radius, weights, [0.16, 0.50, 0.84])
    print(f"  |r| < theta_E  (subhalo inside the Einstein ring)  = "
          f"{weights[radius < samples[:, names.index('theta_E')]].sum() / total:.4f}")
    print(f"  projected radius r = {q50:.4f}  ({q16:.4f}, {q84:.4f})")

    if "theta_E" in names:
        te_lens = samples[:, names.index("theta_E")]
        q16, q50, q84 = weighted_quantile(te_lens, weights, [0.16, 0.50, 0.84])
        print()
        print(f"main-lens theta_E = {q50:+.5f}  ({q16:+.5f}, {q84:+.5f})")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "output/stage_m5.pkl")
