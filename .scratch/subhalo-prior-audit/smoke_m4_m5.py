"""Numerical smoke of the current global-driven M4/PJ-M5 comparison.

Uses a deterministic 32x32 synthetic image, an 8x8 source grid and synthetic
M2/M3 posteriors. No cached data, sampling, plots or output files are touched.
Checks actual shared priors, the log-uniform lambda transform, finite real
likelihoods and the PJ null limit at two different shared-parameter points.
Historical SIS outputs and the removed frozen configuration are not exercised.

Run from the repository root with its Python environment::

    PYTHONPATH=. python .scratch/subhalo-prior-audit/smoke_m4_m5.py
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "examples/pix_src_demo_operator/pipe/subhalo_0946/model.py"
spec = importlib.util.spec_from_file_location("subhalo_smoke_model", MODEL_PATH)
model = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = model
spec.loader.exec_module(model)

from TinyLensGpu.Inference import StagePosterior
from TinyLensGpu.Inference.build_likelihood import make_likelihood
from TinyLensGpu.Inference.build_prior import extract_prior_specs, make_prior_transformation

# Changes affect only this imported module instance, never the pipeline files.
model.NSRC = 8
model.NSUB_PIX = 1
model.COMPARISON_USE_POSITION_LIKELIHOOD = False
model.SUBHALO_POSITION_MODE = "full_image"

mass_names = list(model.SMOOTH_MASS_PRIOR_LIMITS)
centres = dict(theta_E=0.45, gamma=2.0, e1_mass=0.05, e2_mass=0.02,
               center_x_mass=0.0, center_y_mass=0.0, gamma1=0.01, gamma2=-0.01)
offsets = np.linspace(-0.002, 0.002, 9)
stage_m3 = StagePosterior(
    np.asarray([[centres[name] + offset for name in mass_names] for offset in offsets]),
    np.ones(len(offsets)), param_names=mass_names,
)
stage_m2 = StagePosterior(
    np.asarray([[-0.1, 0.8], [0.0, 1.0], [0.1, 1.2]]), np.ones(3),
    param_names=["log_lambda_reg", "adaptive_reg_rho"],
)

y, x = np.mgrid[:32, :32]
x = (x - 15.5) * model.DPIX
y = (y - 15.5) * model.DPIX
image = np.exp(-0.5 * ((np.hypot(x, y) - 0.45) / 0.08) ** 2)
image *= 1.0 + 0.2 * np.cos(np.arctan2(y, x))
noise = np.full_like(image, 0.05)
psf = np.asarray([[0.0, 1.0, 0.0], [1.0, 4.0, 1.0], [0.0, 1.0, 0.0]]) / 8.0
mask = np.zeros(image.shape, dtype=bool)
axis = np.linspace(-0.4, 0.4, model.NSRC)
sx, sy = np.meshgrid(axis, axis)
source_pixels = np.exp(-0.5 * ((sx / 0.15) ** 2 + (sy / 0.15) ** 2)).reshape(-1)
s1_package = dict(
    source_pixels=source_pixels, source_bbox=(-0.4, 0.4, -0.4, 0.4),
    source_x_axis=axis, source_y_axis=axis, n=model.NSRC,
    lambda_best=1.0, log_lambda_best=0.0,
)

likelihoods = {
    tag: model.build_comparison_likelihood(
        image, noise, psf, mask, stage_m2, stage_m3, None, s1_package,
        with_subhalo=with_subhalo,
    )
    for tag, with_subhalo in (("M4", False), ("M5", True))
}
tables = {
    tag: {s.name: (s.prior_type, tuple(s.settings),
                   None if s.limits is None else tuple(s.limits))
          for s in extract_prior_specs(likelihood)}
    for tag, likelihood in likelihoods.items()
}
assert set(tables["M5"]) - set(tables["M4"]) == {
    "b_subhalo", "center_x_subhalo", "center_y_subhalo",
}
assert all(tables["M5"][name] == prior for name, prior in tables["M4"].items())
assert tables["M5"]["b_subhalo"][1] == tuple(model.SUBHALO_B_RANGE)
print("PASS: actual M4/M5 shared priors agree; M5 adds only PJ strength and position")

centre = stage_m2.median("log_lambda_reg")
half = model.LAMBDA_PRIOR_HALF_WIDTH_DEX * np.log(10.0)
values, points, functions, names_by_tag = {}, {}, {}, {}
for tag, likelihood in likelihoods.items():
    prior_fn, specs = make_prior_transformation(likelihood)
    names = [s.name for s in specs]
    cube = np.full((3, len(names)), 0.5)
    cube[:, names.index("log_lambda_reg")] = [0.0, 0.5, 1.0]
    transformed = np.asarray(prior_fn(cube))
    np.testing.assert_allclose(
        transformed[:, names.index("log_lambda_reg")],
        [centre - half, centre, centre + half], rtol=0, atol=1e-6,
    )
    point = transformed[1].copy()
    loglike = make_likelihood(likelihood, vectorized=True)
    value = float(np.asarray(loglike(point[None, :])).reshape(-1)[0])
    assert np.isfinite(value), (tag, value)
    values[tag], points[tag], functions[tag], names_by_tag[tag] = value, point, loglike, names
    print(f"{tag}: lnL={value:.6f}, dimensions={len(names)}")

# Assemble the M5 null point from a valid M5 vector plus all M4 nuisance values.
zero_point = points["M5"].copy()
for name, value in zip(names_by_tag["M4"], points["M4"]):
    zero_point[names_by_tag["M5"].index(name)] = value
zero_point[names_by_tag["M5"].index("b_subhalo")] = 0.0
zero_value = float(np.asarray(functions["M5"](zero_point[None, :])).reshape(-1)[0])
np.testing.assert_allclose(zero_value, values["M4"], rtol=0, atol=1e-3)
assert abs(values["M5"] - zero_value) > 1e-3, "nonzero PJ must perturb this image likelihood"
print(f"M5 b_subhalo=0: lnL={zero_value:.6f}; null difference={zero_value - values['M4']:.3e}")
print(f"Nonzero PJ likelihood change: {values['M5'] - zero_value:+.6f}")

# The decomposition diagnostic also switches PJ off at M5's shifted host point.
# This must match M4 there, not M4 at its original parameter point.
shifted = points["M5"].copy()
shifted[names_by_tag["M5"].index("theta_E")] *= 0.98
shifted[names_by_tag["M5"].index("log_lambda_reg")] += 0.1
shifted_values = dict(zip(names_by_tag["M5"], shifted))
null_at_shifted = np.asarray([shifted_values[name] for name in names_by_tag["M4"]])
smooth_value = float(np.asarray(functions["M4"](null_at_shifted[None, :]))[0])
on_value = float(np.asarray(functions["M5"](shifted[None, :]))[0])
shifted[names_by_tag["M5"].index("b_subhalo")] = 0.0
off_value = float(np.asarray(functions["M5"](shifted[None, :]))[0])
np.testing.assert_allclose(off_value, smooth_value, rtol=0, atol=1e-3)
smooth_gain = smooth_value - values["M4"]
subhalo_gain = on_value - off_value
np.testing.assert_allclose(
    smooth_gain + subhalo_gain, on_value - values["M4"], rtol=0, atol=1e-3,
)
print(f"Shifted host: PJ-off minus M4={off_value - smooth_value:+.3e}")
print(f"Pointwise lnL changes: shared parameters={smooth_gain:+.6f}, "
      f"PJ={subhalo_gain:+.6f}")
print("SMOKE PASSED: shared priors, PJ on/off and pointwise likelihood decomposition")
