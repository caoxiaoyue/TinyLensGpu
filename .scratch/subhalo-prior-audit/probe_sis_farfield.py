"""Is an UNTRUNCATED SIS subhalo degenerate with (theta_E, external shear)?

TinyLensGpu's SIS has a deflection angle of constant magnitude theta_E at every
radius (``PhysicalModel/.../Mass/sis.py``: ``scale = theta_E / radius``).  A
subhalo placed far from the arc therefore still deflects the arc's rays, and its
deflection field over the ring region becomes nearly a constant vector plus a
linear gradient -- exactly the two terms the main lens's theta_E and the external
shear already provide.  This script quantifies how much of the subhalo's
deflection field over the ring lies in that (constant + gradient) subspace, as a
function of the subhalo's projected radius.

Pure numpy; no lens-model or JAX import needed.

Usage::

    python probe_sis_farfield.py
"""

import numpy as np

RING_INNER = 1.20   # inner-ring annulus in the image plane (arcsec; R_E1 = 1.43)
RING_OUTER = 1.60
N_SIDE = 41         # sample grid over the annulus


def ring_points():
    y, x = np.mgrid[-2.0:2.0:complex(N_SIDE), -2.0:2.0:complex(N_SIDE)]
    r = np.hypot(x, y)
    sel = (r >= RING_INNER) & (r <= RING_OUTER)
    return np.stack([x[sel], y[sel]], axis=1)


def sis_deflection(points, centre, theta_e):
    """Deflection of an SIS: theta_e times the unit vector away from centre."""
    d = points - np.asarray(centre)[None, :]
    dist = np.hypot(d[:, 0], d[:, 1])
    return theta_e * d / dist[:, None]


def constant_plus_gradient_fraction(points, field):
    """Fraction of the field's power inside span{1, x, y} per component."""
    design = np.column_stack([np.ones(len(points)), points[:, 0], points[:, 1]])
    projected = np.zeros_like(field)
    for comp in range(2):
        coeff, *_ = np.linalg.lstsq(design, field[:, comp], rcond=None)
        projected[:, comp] = design @ coeff
    total = np.sum(field ** 2)
    residual = np.sum((field - projected) ** 2)
    return 1.0 - residual / total


def main():
    points = ring_points()
    reference = np.array([-0.70744, 1.00661])
    direction = reference / np.linalg.norm(reference)
    theta_e_sub = 0.05

    print("=" * 78)
    print("Untruncated SIS subhalo: how much of its deflection over the arc is")
    print("already expressible as a constant vector + a linear gradient?")
    print(f"  arc annulus: {RING_INNER} <= r <= {RING_OUTER} arcsec, "
          f"{len(points)} sample points")
    print(f"  subhalo theta_E_sub = {theta_e_sub} (only sets the amplitude)")
    print("=" * 78)
    print()
    print("  |p| (arcsec)   fraction in {const, x, y}   fraction NOT (curvature)")
    for radius in (0.5, 0.8, 1.0, 1.32, 1.6, 2.0, 2.5, 3.0, 4.0, 4.5, 6.0):
        centre = direction * radius
        field = sis_deflection(points, centre, theta_e_sub)
        frac = constant_plus_gradient_fraction(points, field)
        print(f"  {radius:12.2f}   {frac:22.4f}   {1.0 - frac:19.4f}")

    print()
    print("Reference: the local control experiment places the subhalo at")
    print(f"  ({reference[0]:+.3f}, {reference[1]:+.3f}), |p| = "
          f"{np.linalg.norm(reference):.3f} arcsec (inner ring, R_E1 = 1.43).")
    print("The full-image posterior puts it at |p| = 2.1 - 4.5 arcsec, where the")
    print("degeneracy with (theta_E, gamma1, gamma2) is essentially complete.")


if __name__ == "__main__":
    main()
