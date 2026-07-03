import numpy as np
import pytest

from TinyLensGpu.Inference import ParamU, StagePosterior
from TinyLensGpu.Inference.stage_posterior import empirical_width
from TinyLensGpu.utils.misc import weighted_quantile


class DummyModule:
    def __init__(self, params):
        self._params = list(params)

    def get_dynamic_params(self):
        return self._params


def _param(name):
    return ParamU(
        name,
        0.0,
        prior_type="uniform",
        prior_settings=[-10.0, 10.0],
        limits=[-10.0, 10.0],
    )


def test_from_likelihood_binds_schema_and_normalizes_weights():
    module = DummyModule([_param("theta_E"), _param("e1_mass")])
    samples = np.array([[1.0, -0.2], [2.0, 0.2], [3.0, 0.4]])
    weights = np.array([1.0, 2.0, 1.0])

    stage = StagePosterior.from_likelihood(module, samples, weights, log_z=-12.5)

    assert stage.param_names == ["theta_E", "e1_mass"]
    assert np.isclose(stage.weights.sum(), 1.0)
    assert stage.log_z == -12.5
    assert stage.median("theta_E") == pytest.approx(
        weighted_quantile(samples[:, 0], stage.weights, 0.5)
    )


def test_from_schema_rehydrates_without_likelihood():
    samples = np.array([[0.0, 1.0], [2.0, 3.0]])
    weights = np.array([0.25, 0.75])

    stage = StagePosterior.from_schema(
        samples,
        weights,
        param_names=["x", "y"],
        log_z=3.0,
    )

    assert stage.likelihood is None
    assert stage.param_names == ["x", "y"]
    assert stage.medians() == {
        "x": pytest.approx(weighted_quantile(samples[:, 0], stage.weights, 0.5)),
        "y": pytest.approx(weighted_quantile(samples[:, 1], stage.weights, 0.5)),
    }


def test_constructor_rejects_shape_mismatch():
    module = DummyModule([_param("x"), _param("y")])

    with pytest.raises(ValueError, match="column count"):
        StagePosterior.from_likelihood(
            module,
            np.ones((4, 1)),
            np.ones(4),
        )


def test_constructor_rejects_duplicate_names():
    module = DummyModule([_param("x"), _param("x")])

    with pytest.raises(ValueError, match="duplicates"):
        StagePosterior.from_likelihood(
            module,
            np.ones((4, 2)),
            np.ones(4),
        )


def test_missing_name_raises_keyerror_with_available_names():
    stage = StagePosterior.from_schema(
        np.ones((3, 1)),
        np.ones(3),
        param_names=["x"],
    )

    with pytest.raises(KeyError, match="Available"):
        stage.median("missing")


def test_summary_methods_match_weighted_values():
    samples = np.array([[0.0], [2.0], [4.0]])
    weights = np.array([1.0, 2.0, 1.0])
    stage = StagePosterior.from_schema(samples, weights, param_names=["x"])

    expected_median = weighted_quantile(samples[:, 0], stage.weights, 0.5)
    assert stage.median("x") == pytest.approx(expected_median)
    assert stage.std("x") == pytest.approx(np.sqrt(2.0))
    assert stage.median_std("x") == pytest.approx((expected_median, np.sqrt(2.0)))
    assert stage.medians()["x"] == pytest.approx(expected_median)


def test_fixed_returns_static_paramu_with_target_name():
    stage = StagePosterior.from_schema(
        np.array([[0.1], [0.3], [0.5]]),
        np.ones(3),
        param_names=["center_x_mass"],
    )

    param = stage.fixed("center_x_mass", target="center_x")

    assert isinstance(param, ParamU)
    assert param.name == "center_x"
    assert float(param.value) == pytest.approx(stage.median("center_x_mass"))
    assert param.static is True
    assert param.dynamic is False


def test_gaussian_matches_conservative_sigma_rule_and_is_dynamic():
    samples = np.array([[1.0], [1.2], [1.4]])
    weights = np.ones(3)
    stage = StagePosterior.from_schema(
        samples,
        weights,
        param_names=["theta_E"],
        factor_std=1.0,
    )

    param = stage.gaussian(
        "theta_E",
        model="EPL",
        attr="theta_E",
        limits=[0.0, 5.0],
    )
    med = stage.median("theta_E")
    _, rel_width = empirical_width("EPL", "theta_E")
    expected_sigma = max(stage.std("theta_E"), rel_width * abs(med))

    assert param.name == "theta_E"
    assert param.prior_type == "gaussian"
    assert param.prior_settings == pytest.approx([med, expected_sigma])
    assert param.limits == [0.0, 5.0]
    assert param.dynamic is True
    assert param.static is False


def test_gaussian_unknown_empirical_width_key_raises():
    stage = StagePosterior.from_schema(
        np.ones((3, 1)),
        np.ones(3),
        param_names=["x"],
    )

    with pytest.raises(KeyError):
        stage.gaussian("x", model="Unknown", attr="x")


def test_schema_payload_round_trip():
    module = DummyModule([_param("x")])
    stage = StagePosterior.from_likelihood(
        module,
        np.array([[1.0], [2.0]]),
        np.ones(2),
        log_z=4.0,
    )

    payload = stage.cache_payload()
    restored = StagePosterior.from_schema(
        payload["samples"],
        payload["weights"],
        prior_specs=payload["prior_specs"],
        log_z=payload["log_z"],
    )

    assert restored.param_names == ["x"]
    assert restored.median("x") == pytest.approx(stage.median("x"))
    assert restored.log_z == 4.0
