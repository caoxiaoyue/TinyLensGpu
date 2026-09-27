"""Is the full-image subhalo-position posterior multi-modal?

J0946+1006 is a double-Einstein-ring system (inner ring R_E1 = 1.43", outer ring
near 2.0").  A subhalo can perturb either ring, so the position posterior may
carry one mode per ring; a localised prior picks one of them.  This script
histograms the stored subhalo-position samples of a stage-M5 cache and reports
modes with their probability mass, projected radius and PJ strength b_subhalo.

No JAX: reads current PJ caches and their saved StagePosterior prior specs.
Historical SIS posterior outputs are not inputs to this current-PJ summary.

Usage::

    python probe_position_modes.py output/stage_m5.pkl [more...]
"""

import pickle
import sys

import numpy as np

INNER_RING = 1.43   # R_E1, z_s1 = 0.609 (SLACS VI / Gavazzi et al. 2008)
OUTER_RING = 2.10   # R_E2, second source (Collett & Smith 2020 give 1.4 and 2.1)


def load(path):
    with open(path, "rb") as handle:
        payload = pickle.load(handle)
    names = list(payload["param_names"])
    cols = {n: np.asarray(payload["samples"], dtype=float)[:, i]
            for i, n in enumerate(names)}
    return payload, cols, np.asarray(payload["weights"], dtype=float)


def peaks(xs, ys, ws, x_lo, x_hi, y_lo, y_hi, bin_size=0.25, min_weight=0.02):
    """Return weighted 2D density peaks (greedy, with a suppression radius)."""
    nx = max(4, int(round((x_hi - x_lo) / bin_size)))
    ny = max(4, int(round((y_hi - y_lo) / bin_size)))
    hist, xe, ye = np.histogram2d(xs, ys, bins=(nx, ny), weights=ws,
                                  range=((x_lo, x_hi), (y_lo, y_hi)))
    total = hist.sum()
    hist = hist / total
    found = []
    work = hist.copy()
    for _ in range(6):
        idx = np.unravel_index(np.argmax(work), work.shape)
        height = work[idx]
        if height < min_weight:
            break
        xc = 0.5 * (xe[idx[0]] + xe[idx[0] + 1])
        yc = 0.5 * (ye[idx[1]] + ye[idx[1] + 1])
        radius = bin_size * 1.5
        near = np.hypot(xs - xc, ys - yc) < radius
        found.append(dict(x=xc, y=yc, height=height,
                          mass=float(ws[near].sum() / ws.sum()),
                          particle=float(near.mean())))
        # Suppress this peak: zero the bins inside the suppression radius.
        gx = (xe[:-1] + 0.5 * np.diff(xe) - xc)
        gy = (ye[:-1] + 0.5 * np.diff(ye) - yc)
        mask = np.hypot(gx[:, None], gy[None, :]) < radius
        work[mask] = 0.0
    return found


def main(paths):
    for path in paths:
        payload, cols, ws = load(path)
        extra = payload.get("extra", {})
        role = extra.get("comparison_role", "?")
        pos = extra["subhalo_position"]
        prior_specs = {s["name"]: s for s in payload["stage_schema"]["prior_specs"]}
        x_spec, y_spec = (prior_specs[f"center_{axis}_subhalo"] for axis in ("x", "y"))
        xs, ys = cols["center_x_subhalo"], cols["center_y_subhalo"]
        strength = cols["b_subhalo"]

        print("=" * 76)
        print(f"{path}")
        print(f"  role = {role}   log Z = {payload['log_z']:+.3f}   "
              f"position mode = {pos.get('mode')}")
        print("=" * 76)

        r = np.hypot(xs, ys)
        q16, q50, q84 = np.percentile(r, [16, 50, 84])
        print(f"  projected radius r = {q50:.3f}  ({q16:.3f}, {q84:.3f}) arcsec")
        print(f"  fraction of weight with r < {INNER_RING}: "
              f"{ws[(r < INNER_RING)].sum() / ws.sum():.3f}")
        print(f"  fraction of weight with r > {OUTER_RING - 0.2}: "
              f"{ws[(r > OUTER_RING - 0.2)].sum() / ws.sum():.3f}")
        q16, q50, q84 = np.percentile(strength, [16, 50, 84])
        print(f"  PJ b_sub = {q50:.4f}  ({q16:.4f}, {q84:.4f}) [unweighted quantiles]")

        near_outer = r > OUTER_RING - 0.35
        if near_outer.any():
            sub_w = ws[near_outer]
            q16, q50, q84 = np.percentile(strength[near_outer], [16, 50, 84])
            print(f"  if r > {OUTER_RING - 0.35:.2f} (outer-ring mode): "
                  f"weight = {sub_w.sum() / ws.sum():.3f}, "
                  f"b_sub = {q50:.4f} ({q16:.4f}, {q84:.4f}) [unweighted quantiles]")
        near_inner = r < INNER_RING
        if near_inner.any():
            sub_w = ws[near_inner]
            q16, q50, q84 = np.percentile(strength[near_inner], [16, 50, 84])
            print(f"  if r < {INNER_RING:.2f} (inner-ring mode):     "
                  f"weight = {sub_w.sum() / ws.sum():.3f}, "
                  f"b_sub = {q50:.4f} ({q16:.4f}, {q84:.4f}) [unweighted quantiles]")

        if x_spec["prior_type"] == y_spec["prior_type"] == "uniform":
            print()
            print("  2D position peaks (bin = 0.25 arcsec):")
            for i, peak in enumerate(peaks(
                    xs, ys, ws, *x_spec["settings"], *y_spec["settings"]), 1):
                rr = float(np.hypot(peak["x"], peak["y"]))
                ring = ("inner ring" if abs(rr - INNER_RING) < 0.35 else
                        "outer ring" if abs(rr - OUTER_RING) < 0.35 else "-")
                print(f"    {i}. ({peak['x']:+.3f}, {peak['y']:+.3f})"
                      f"  r = {rr:.3f}  [{ring:<10}]"
                      f"  density = {peak['height']:.4f}"
                      f"  mass = {peak['mass']:.3f}")
        print()


if __name__ == "__main__":
    main(sys.argv[1:] or ["output/stage_m5.pkl"])
