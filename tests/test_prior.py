"""Tests for all prior types defined in param_u and build_prior.

Covers transform math, describe formatting, JAX compatibility, and
end-to-end wiring for uniform, log_uniform, gaussian, and truncated_gaussian.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from TinyLensGpu.Inference.param_u import ParamU
from TinyLensGpu.Inference.build_prior import (
    PriorSpec,
    extract_prior_specs,
    make_prior_transformation,
)


# ================================================================
#  PriorSpec.transform() — unit tests
# ================================================================

class TestPriorSpecTransform:
    """Direct unit tests for PriorSpec.transform()."""

    # -- helpers --

    @staticmethod
    def _spec(**kw):
        return PriorSpec("x", **kw)

    # -- truncated_gaussian --

    @pytest.mark.parametrize(
        "mean, std, low, high, u, expected_range",
        [
            # median of unit cube maps near the mean
            (1.0, 0.2, 0.5, 2.0, 0.5, (0.95, 1.05)),
            # u=0 maps near lower bound
            (1.0, 0.2, 0.5, 2.0, 0.0, (0.49, 0.70)),
            # u=1 maps near upper bound
            (1.0, 0.2, 0.5, 2.0, 1.0, (1.30, 2.01)),
            # symmetric truncation — median should equal mean exactly
            (0.0, 1.0, -2.0, 2.0, 0.5, (-0.05, 0.05)),
            # narrow truncation far from mean
            (10.0, 1.0, 9.5, 10.5, 0.5, (9.90, 10.10)),
        ],
    )
    def test_truncated_gaussian_bounds(self, mean, std, low, high, u, expected_range):
        spec = self._spec(
            prior_type="truncated_gaussian",
            settings=(mean, std),
            limits=(low, high),
        )
        val = float(spec.transform(jnp.array(u)))
        lo, hi = expected_range
        assert lo <= val <= hi, f"u={u}: expected {val} in [{lo}, {hi}]"

    def test_truncated_gaussian_values_never_exceed_limits(self):
        """All outputs must stay within [low, high] for any u in [0,1]."""
        spec = self._spec(
            prior_type="truncated_gaussian",
            settings=(1.0, 0.5),
            limits=(0.2, 3.0),
        )
        u = jnp.linspace(0.0, 1.0, 1001)
        vals = spec.transform(u)
        assert jnp.all(vals >= 0.2), f"min = {float(jnp.min(vals))}"
        assert jnp.all(vals <= 3.0), f"max = {float(jnp.max(vals))}"

    def test_truncated_gaussian_monotonic(self):
        """Transform must be monotonically increasing in u."""
        spec = self._spec(
            prior_type="truncated_gaussian",
            settings=(1.0, 0.3),
            limits=(0.4, 1.6),
        )
        u = jnp.linspace(0.0, 1.0, 200)
        vals = spec.transform(u)
        diffs = jnp.diff(vals)
        assert jnp.all(diffs >= 0.0), f"non-monotonic at {float(jnp.min(diffs))}"

    # -- boundary condition: truncated =~ gaussian for wide bounds --

    def test_truncated_approximates_gaussian_wide_bounds(self):
        """With limits far from the mean, truncated ≈ untruncated Gaussian."""
        trunc = self._spec(
            prior_type="truncated_gaussian",
            settings=(0.0, 1.0),
            limits=(-100.0, 100.0),
        )
        gauss = self._spec(
            prior_type="gaussian",
            settings=(0.0, 1.0),
        )
        u = jnp.linspace(0.01, 0.99, 200)
        max_diff = float(jnp.max(jnp.abs(trunc.transform(u) - gauss.transform(u))))
        assert max_diff < 0.05, f"max difference = {max_diff}"

    # -- JAX compatibility --

    def test_truncated_gaussian_is_jittable(self):
        spec = self._spec(
            prior_type="truncated_gaussian",
            settings=(1.0, 0.2),
            limits=(0.5, 2.0),
        )
        u = jnp.array([0.0, 0.5, 1.0])
        eager = spec.transform(u)
        jitted = jax.jit(spec.transform)(u)
        # Values agree within 0.05 (JIT may differ slightly at clamped boundaries)
        assert jnp.allclose(eager, jitted, atol=0.05), f"{eager} vs {jitted}"

    def test_truncated_gaussian_gradient_is_finite(self):
        """Transform should be differentiable w.r.t. u everywhere in (0,1)."""
        spec = self._spec(
            prior_type="truncated_gaussian",
            settings=(1.0, 0.2),
            limits=(0.5, 2.0),
        )
        grad_fn = jax.grad(lambda u: jnp.sum(spec.transform(u)))
        u = jnp.array([0.3, 0.5, 0.7])
        grad = grad_fn(u)
        assert jnp.all(jnp.isfinite(grad)), f"gradient = {grad}"
        assert jnp.all(grad > 0.0), "transform must be strictly increasing"

    # -- uniform --

    @pytest.mark.parametrize(
        "low, high, u, expected",
        [
            (0.0, 2.0, 0.0, 0.0),     # u=0 → min
            (0.0, 2.0, 1.0, 2.0),     # u=1 → max
            (0.0, 2.0, 0.5, 1.0),     # u=0.5 → midpoint
            (-5.0, 5.0, 0.25, -2.5),  # quarter
            (1.0, 3.0, 0.75, 2.5),    # three-quarters
        ],
    )
    def test_uniform_exact_boundaries(self, low, high, u, expected):
        """Uniform prior: u maps linearly from min to max."""
        spec = self._spec(prior_type="uniform", settings=(low, high))
        val = float(spec.transform(jnp.array(u)))
        assert val == pytest.approx(expected, abs=0.02)

    def test_uniform_monotonic(self):
        spec = self._spec(prior_type="uniform", settings=(0.0, 10.0))
        u = jnp.linspace(0.0, 1.0, 200)
        diffs = jnp.diff(spec.transform(u))
        assert jnp.all(diffs >= 0.0)

    def test_uniform_is_jittable(self):
        spec = self._spec(prior_type="uniform", settings=(0.0, 2.0))
        u = jnp.linspace(0.0, 1.0, 50)
        eager = spec.transform(u)
        jitted = jax.jit(spec.transform)(u)
        assert jnp.allclose(eager, jitted)

    # -- log_uniform --

    @pytest.mark.parametrize(
        "low, high, u, expected_range",
        [
            (0.1, 10.0, 0.0, (0.10, 0.11)),   # u=0 → near low
            (0.1, 10.0, 1.0, (9.90, 10.01)),   # u=1 → near high
            (0.1, 10.0, 0.5, (0.95, 1.05)),    # u=0.5 → geometric midpoint
        ],
    )
    def test_log_uniform_boundaries(self, low, high, u, expected_range):
        """Log-uniform prior: u maps log-linearly from min to max."""
        spec = self._spec(prior_type="log_uniform", settings=(low, high))
        val = float(spec.transform(jnp.array(u)))
        lo, hi = expected_range
        assert lo <= val <= hi, f"u={u}: expected {val} in [{lo}, {hi}]"

    def test_log_uniform_values_positive(self):
        """Log-uniform must always produce positive values."""
        spec = self._spec(prior_type="log_uniform", settings=(0.001, 1000.0))
        u = jnp.linspace(0.0, 1.0, 200)
        vals = spec.transform(u)
        assert jnp.all(vals > 0.0)

    def test_log_uniform_monotonic(self):
        spec = self._spec(prior_type="log_uniform", settings=(0.1, 10.0))
        u = jnp.linspace(0.0, 1.0, 200)
        diffs = jnp.diff(spec.transform(u))
        assert jnp.all(diffs >= 0.0)

    def test_log_uniform_is_jittable(self):
        spec = self._spec(prior_type="log_uniform", settings=(0.1, 10.0))
        u = jnp.linspace(0.01, 0.99, 50)
        eager = spec.transform(u)
        jitted = jax.jit(spec.transform)(u)
        assert jnp.allclose(eager, jitted)

    # -- gaussian --

    @pytest.mark.parametrize(
        "mean, std, u, expected_range",
        [
            (0.0, 1.0, 0.5, (-0.01, 0.01)),    # u=0.5 → mean
            (5.0, 2.0, 0.5, (4.98, 5.02)),      # non-zero mean
            (0.0, 1.0, 0.025, (-2.1, -1.8)),    # u near 0 → left tail
            (0.0, 1.0, 0.975, (1.8, 2.1)),      # u near 1 → right tail
        ],
    )
    def test_gaussian_quantiles(self, mean, std, u, expected_range):
        """Gaussian prior: inverse-CDF maps correctly at key quantiles."""
        spec = self._spec(prior_type="gaussian", settings=(mean, std))
        val = float(spec.transform(jnp.array(u)))
        lo, hi = expected_range
        assert lo <= val <= hi, f"u={u}: expected {val} in [{lo}, {hi}]"

    def test_gaussian_with_limits_clips(self):
        """Gaussian with limits clips at boundaries."""
        spec = self._spec(
            prior_type="gaussian",
            settings=(0.0, 1.0),
            limits=(-1.5, 1.5),
        )
        u = jnp.linspace(0.0, 1.0, 500)
        vals = spec.transform(u)
        assert jnp.all(vals >= -1.5)
        assert jnp.all(vals <= 1.5)

    def test_gaussian_monotonic(self):
        spec = self._spec(prior_type="gaussian", settings=(0.0, 1.0))
        u = jnp.linspace(0.01, 0.99, 200)
        diffs = jnp.diff(spec.transform(u))
        assert jnp.all(diffs >= 0.0)

    def test_gaussian_is_jittable(self):
        spec = self._spec(prior_type="gaussian", settings=(1.0, 0.5))
        u = jnp.linspace(0.01, 0.99, 50)
        eager = spec.transform(u)
        jitted = jax.jit(spec.transform)(u)
        assert jnp.allclose(eager, jitted)

    def test_gaussian_gradient_is_finite(self):
        spec = self._spec(prior_type="gaussian", settings=(1.0, 0.2))
        grad_fn = jax.grad(lambda u: jnp.sum(spec.transform(u)))
        u = jnp.array([0.3, 0.5, 0.7])
        grad = grad_fn(u)
        assert jnp.all(jnp.isfinite(grad))
        assert jnp.all(grad > 0.0)

    # -- all types: gradient positivity --

    @pytest.mark.parametrize("prior_type,settings,limits", [
        ("uniform", (0.0, 2.0), None),
        ("log_uniform", (0.1, 10.0), None),
        ("gaussian", (0.0, 1.0), None),
        ("truncated_gaussian", (1.0, 0.2), (0.5, 2.0)),
    ])
    def test_all_types_strictly_increasing(self, prior_type, settings, limits):
        """Every prior transform must be strictly increasing in u."""
        spec = self._spec(prior_type=prior_type, settings=settings, limits=limits)
        u = jnp.linspace(0.01, 0.99, 200)
        diffs = jnp.diff(spec.transform(u))
        assert jnp.all(diffs > 0.0), f"{prior_type}: gradient not strictly positive"

    # -- regression: existing prior types unchanged --

    @pytest.mark.parametrize(
        "prior_type, settings, limits, u, expected",
        [
            ("uniform", (0.0, 2.0), None, 0.5, 1.0),
            ("log_uniform", (0.1, 10.0), None, 0.5, 1.0),
            ("gaussian", (1.0, 0.2), None, 0.5, 1.0),  # u=0.5 → mean
        ],
    )
    def test_existing_prior_types_unchanged(
        self, prior_type, settings, limits, u, expected
    ):
        spec = self._spec(prior_type=prior_type, settings=settings, limits=limits)
        val = float(spec.transform(jnp.array(u)))
        assert val == pytest.approx(expected, abs=0.02)

    def test_unsupported_prior_type_raises(self):
        spec = self._spec(prior_type="gaussian", settings=(0.0, 1.0))
        # Mutate the frozen dataclass field so we can test the error branch
        object.__setattr__(spec, "prior_type", "cauchy")
        with pytest.raises(ValueError, match="Unsupported prior type"):
            spec.transform(jnp.array(0.5))


# ================================================================
#  PriorSpec.describe()
# ================================================================

class TestPriorSpecDescribe:
    """Unit tests for PriorSpec.describe()."""

    def test_truncated_gaussian_describe(self):
        spec = PriorSpec("x", "truncated_gaussian", (1.0, 0.2), limits=(0.5, 2.0))
        desc = spec.describe()
        assert desc == "TN(1.00, 0.20, [0.50, 2.00])"

    def test_truncated_gaussian_describe_no_limits_suffix(self):
        """Limits for truncated_gaussian are intrinsic; no redundant suffix."""
        spec = PriorSpec("x", "truncated_gaussian", (5.0, 1.0), limits=(3.0, 7.0))
        desc = spec.describe()
        assert "limits=" not in desc
        assert desc.startswith("TN(")

    def test_gaussian_describe_still_appends_limits(self):
        """Non-truncated priors should still show limits= suffix when limits exist."""
        spec = PriorSpec("x", "gaussian", (0.0, 1.0), limits=(-2.0, 2.0))
        desc = spec.describe()
        assert "limits=" in desc
        assert desc.startswith("N(")

    @pytest.mark.parametrize(
        "prior_type, settings, limits, startswith",
        [
            ("uniform", (0.0, 2.0), None, "["),
            ("log_uniform", (0.1, 10.0), None, "["),
            ("gaussian", (0.0, 1.0), None, "N("),
            ("truncated_gaussian", (0.0, 1.0), (0.0, 1.0), "TN("),
        ],
    )
    def test_all_prior_types_describe_prefix(
        self, prior_type, settings, limits, startswith
    ):
        spec = PriorSpec("x", prior_type, settings, limits=limits)
        assert spec.describe().startswith(startswith)


# ================================================================
#  ParamU — construction and metadata
# ================================================================

class TestParamU:
    """Tests for ParamU construction across all prior types."""

    # -- truncated_gaussian --

    def test_construction_truncated_gaussian(self):
        p = ParamU(
            "theta_E",
            1.0,
            prior_type="truncated_gaussian",
            prior_settings=[1.0, 0.2],
            limits=[0.5, 2.0],
        )
        assert p.prior_type == "truncated_gaussian"
        assert p.prior_settings == [1.0, 0.2]
        assert p.limits == [0.5, 2.0]

    # -- uniform --

    def test_construction_uniform(self):
        p = ParamU("a", 0.5, prior_type="uniform",
                   prior_settings=[0.0, 1.0], limits=[0.0, 1.0])
        assert p.prior_type == "uniform"
        assert p.prior_settings == [0.0, 1.0]
        assert p.limits == [0.0, 1.0]

    def test_construction_uniform_defaults(self):
        """Default prior_type is uniform."""
        p = ParamU("x")
        assert p.prior_type == "uniform"
        assert p.prior_settings is None
        assert p.limits is None

    # -- log_uniform --

    def test_construction_log_uniform(self):
        p = ParamU("b", 1.0, prior_type="log_uniform",
                   prior_settings=[0.1, 10.0], limits=[0.1, 10.0])
        assert p.prior_type == "log_uniform"
        assert p.prior_settings == [0.1, 10.0]

    # -- gaussian --

    def test_construction_gaussian(self):
        p = ParamU("c", 0.0, prior_type="gaussian",
                   prior_settings=[0.0, 1.0], limits=[-5.0, 5.0])
        assert p.prior_type == "gaussian"
        assert p.prior_settings == [0.0, 1.0]

    def test_construction_gaussian_no_limits(self):
        """Gaussian supports None limits."""
        p = ParamU("d", prior_type="gaussian", prior_settings=[0.0, 1.0])
        assert p.prior_type == "gaussian"
        assert p.limits is None

    # -- repr --

    @pytest.mark.parametrize("prior_type", [
        "uniform", "log_uniform", "gaussian", "truncated_gaussian",
    ])
    def test_repr_includes_prior_type(self, prior_type):
        p = ParamU("x", prior_type=prior_type,
                   prior_settings=[0.0, 1.0], limits=[-2.0, 2.0])
        r = repr(p)
        assert prior_type in r
        assert "ParamU(" in r


# ================================================================
#  extract_prior_specs / make_prior_transformation — wiring
# ================================================================

class DummyModule:
    """Minimal module that exposes dynamic params for extract_prior_specs."""

    def __init__(self, params):
        self._params = list(params)

    def get_dynamic_params(self):
        return self._params


class TestExtractPriorSpecs:
    """Integration tests for extract_prior_specs across all prior types."""

    @pytest.mark.parametrize("prior_type,settings,limits", [
        ("uniform", [0.0, 1.0], [0.0, 1.0]),
        ("log_uniform", [0.1, 10.0], [0.1, 10.0]),
        ("gaussian", [0.0, 1.0], [-5.0, 5.0]),
        ("truncated_gaussian", [2.0, 0.5], [0.0, 5.0]),
    ])
    def test_extracts_single_spec(self, prior_type, settings, limits):
        p = ParamU("p", prior_type=prior_type,
                   prior_settings=settings, limits=limits)
        module = DummyModule([p])
        specs = extract_prior_specs(module)
        assert len(specs) == 1
        s = specs[0]
        assert s.name == "p"
        assert s.prior_type == prior_type
        assert s.settings == tuple(settings)
        assert s.limits == tuple(limits)

    def test_extracts_mixed_prior_types(self):
        p1 = ParamU("u", prior_type="uniform",
                    prior_settings=[0.0, 1.0], limits=[0.0, 1.0])
        p2 = ParamU("l", prior_type="log_uniform",
                    prior_settings=[0.1, 10.0], limits=[0.1, 10.0])
        p3 = ParamU("g", prior_type="gaussian",
                    prior_settings=[0.0, 1.0], limits=[-5.0, 5.0])
        p4 = ParamU("t", prior_type="truncated_gaussian",
                    prior_settings=[3.0, 0.3], limits=[2.0, 4.0])
        module = DummyModule([p1, p2, p3, p4])
        specs = extract_prior_specs(module)
        types = {s.name: s.prior_type for s in specs}
        assert types == {
            "u": "uniform",
            "l": "log_uniform",
            "g": "gaussian",
            "t": "truncated_gaussian",
        }

    @pytest.mark.parametrize("prior_type", [
        "uniform", "log_uniform", "gaussian", "truncated_gaussian",
    ])
    def test_missing_limits_raises(self, prior_type):
        p = ParamU("x", prior_type=prior_type, prior_settings=[1.0, 0.2])
        module = DummyModule([p])
        with pytest.raises(ValueError, match="limits must be a tuple of length 2"):
            extract_prior_specs(module)

    @pytest.mark.parametrize("prior_type", [
        "uniform", "log_uniform", "gaussian", "truncated_gaussian",
    ])
    def test_missing_prior_settings_raises(self, prior_type):
        p = ParamU("x", prior_type=prior_type, limits=[0.0, 1.0])
        module = DummyModule([p])
        with pytest.raises(ValueError, match="prior_settings must be a tuple"):
            extract_prior_specs(module)

    def test_no_dynamic_params_raises(self):
        module = DummyModule([])
        with pytest.raises(ValueError, match="no dynamic parameters"):
            extract_prior_specs(module)


class TestMakePriorTransformation:
    """End-to-end tests for make_prior_transformation."""

    @pytest.mark.parametrize("prior_type,settings,limits,check_positivity", [
        ("uniform", [0.0, 2.0], [0.0, 2.0], False),
        ("log_uniform", [0.1, 10.0], [0.1, 10.0], True),
        ("gaussian", [0.0, 1.0], [-5.0, 5.0], False),
        ("truncated_gaussian", [1.0, 0.2], [0.5, 2.0], False),
    ])
    def test_transform_produces_values_within_limits(
        self, prior_type, settings, limits, check_positivity
    ):
        p = ParamU("x", prior_type=prior_type,
                   prior_settings=settings, limits=limits)
        module = DummyModule([p])
        transform, specs = make_prior_transformation(module)
        assert len(specs) == 1
        assert specs[0].prior_type == prior_type

        rng = np.random.default_rng(42)
        u_batch = jnp.asarray(rng.uniform(0, 1, (2000, 1)))
        theta = transform(u_batch)
        assert theta.shape == (2000, 1)
        assert jnp.all(theta >= limits[0]), f"min below {limits[0]}: {float(jnp.min(theta))}"
        assert jnp.all(theta <= limits[1]), f"max above {limits[1]}: {float(jnp.max(theta))}"
        if check_positivity:
            assert jnp.all(theta > 0.0)

    @pytest.mark.parametrize("prior_type,settings,limits,expected_range", [
        ("uniform", [0.0, 2.0], [0.0, 2.0], (0.98, 1.02)),
        ("log_uniform", [0.1, 10.0], [0.1, 10.0], (0.95, 1.05)),
        ("gaussian", [0.0, 1.0], [-5.0, 5.0], (-0.02, 0.02)),
        ("truncated_gaussian", [0.0, 1.0], [-2.0, 2.0], (-0.02, 0.02)),
    ])
    def test_transform_median_behavior(
        self, prior_type, settings, limits, expected_range
    ):
        """u=0.5 should map to the 'center' of each prior."""
        p = ParamU("x", prior_type=prior_type,
                   prior_settings=settings, limits=limits)
        module = DummyModule([p])
        transform, _ = make_prior_transformation(module)
        theta = float(transform(jnp.array([[0.5]]))[0, 0])
        lo, hi = expected_range
        assert lo <= theta <= hi, f"{prior_type}: theta={theta} not in [{lo}, {hi}]"

    @pytest.mark.parametrize("prior_type,settings,limits", [
        ("uniform", [0.0, 2.0], [0.0, 2.0]),
        ("log_uniform", [0.1, 10.0], [0.1, 10.0]),
        ("gaussian", [0.0, 1.0], [-5.0, 5.0]),
        ("truncated_gaussian", [0.0, 1.0], [-3.0, 3.0]),
    ])
    def test_transform_is_jittable(self, prior_type, settings, limits):
        p = ParamU("x", prior_type=prior_type,
                   prior_settings=settings, limits=limits)
        module = DummyModule([p])
        transform, _ = make_prior_transformation(module)
        jitted = jax.jit(transform)
        u = jnp.array([[0.5]])
        eager_val = transform(u)
        jit_val = jitted(u)
        assert jnp.allclose(eager_val, jit_val, atol=0.05)

    def test_transform_with_multiple_params_mixed_types(self):
        """End-to-end with all four prior types in one module."""
        params = [
            ParamU("u", prior_type="uniform",
                   prior_settings=[0.0, 1.0], limits=[0.0, 1.0]),
            ParamU("l", prior_type="log_uniform",
                   prior_settings=[0.1, 10.0], limits=[0.1, 10.0]),
            ParamU("g", prior_type="gaussian",
                   prior_settings=[0.0, 1.0], limits=[-5.0, 5.0]),
            ParamU("t", prior_type="truncated_gaussian",
                   prior_settings=[1.0, 0.3], limits=[0.2, 2.0]),
        ]
        module = DummyModule(params)
        transform, specs = make_prior_transformation(module)
        assert len(specs) == 4

        rng = np.random.default_rng(123)
        u_batch = jnp.asarray(rng.uniform(0, 1, (500, 4)))
        theta = transform(u_batch)
        assert theta.shape == (500, 4)
        # Per-column bounds
        bounds = [(0.0, 1.0), (0.1, 10.0), (-5.0, 5.0), (0.2, 2.0)]
        for i, (lo, hi) in enumerate(bounds):
            col = theta[:, i]
            assert jnp.all(col >= lo), f"col {i} ({specs[i].name}): min={float(jnp.min(col))} < {lo}"
            assert jnp.all(col <= hi), f"col {i} ({specs[i].name}): max={float(jnp.max(col))} > {hi}"
