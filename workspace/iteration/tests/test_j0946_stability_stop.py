"""Windowed stability must observe the actual optimizer's physical fields."""

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest


PATH = Path(__file__).resolve().parents[1] / "stability_stop.py"
SPEC = importlib.util.spec_from_file_location("j0946_stability_stop_test", PATH)
stop = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = stop
SPEC.loader.exec_module(stop)


def fields(source=1., potential=.001, kappa=.1):
    return dict(source=np.atleast_1d(source), potential=np.atleast_1d(potential),
                kappa=np.atleast_1d(kappa))


def test_requires_complete_windows_and_consecutive_checks():
    monitor = stop.StabilityMonitor(stop.StabilityPolicy(window=4, check_every=2, consecutive=3))
    verdicts = [monitor.observe(i, 5., fields()) for i in range(9)]
    assert not any(verdicts[:8])
    assert verdicts[8]
    assert monitor.first_hit["iteration"] == 8


@pytest.mark.parametrize("component", ("source", "potential", "kappa"))
def test_objective_plateau_does_not_hide_a_drifting_physical_block(component):
    monitor = stop.StabilityMonitor(stop.StabilityPolicy(window=4, check_every=1, consecutive=1))
    for i in range(8):
        state = fields(source=1e6)  # A joint-vector norm would hide potential/kappa drift.
        state[component] = np.array([1.+i*.01])
        assert not monitor.observe(i, 5., state)


def test_equal_endpoints_do_not_hide_window_excursions():
    monitor = stop.StabilityMonitor(stop.StabilityPolicy(window=4, check_every=1, consecutive=1))
    for i in range(5):
        assert not monitor.observe(i, 5., fields(kappa=.1 if i % 2 == 0 else .2))
    assert not monitor.records[-1]["kappa_max_satisfied"]


def test_objective_checks_every_accepted_value_between_field_samples():
    monitor = stop.StabilityMonitor(stop.StabilityPolicy(window=4, check_every=2, consecutive=1))
    for i in range(5):
        assert not monitor.observe(i, 6. if i == 1 else 5., fields())
    assert not monitor.records[-1]["objective_satisfied"]


def test_nonfinite_window_cannot_qualify_and_zero_fields_are_valid():
    monitor = stop.StabilityMonitor(stop.StabilityPolicy(window=2, check_every=1, consecutive=1))
    assert not monitor.observe(0, 5., fields(kappa=np.nan))
    assert not monitor.observe(1, 5., fields(kappa=0.))
    assert not monitor.observe(2, 5., fields(kappa=0.))
    assert monitor.observe(3, 5., fields(kappa=0.))


def test_a_local_kappa_change_cannot_hide_in_a_large_field_norm():
    monitor = stop.StabilityMonitor(stop.StabilityPolicy(window=2, check_every=1, consecutive=1,
                                                       solution_rtol=.01))
    for i in range(3):
        kappa = np.ones(10000)
        kappa[0] += .1*i
        assert not monitor.observe(i, 5., fields(kappa=kappa))
    assert monitor.records[-1]["kappa_L2_satisfied"]
    assert not monitor.records[-1]["kappa_max_satisfied"]


def test_real_scipy_callback_stops_at_the_checked_state_without_relabeling_success():
    diagonal = np.geomspace(1., 1e4, 30)

    def objective(x):
        return .5*np.dot(x, diagonal*x), diagonal*x

    def sample(x):
        return fields(source=x, potential=x*.01, kappa=x*.1)

    policy = stop.StabilityPolicy(window=2, check_every=1, consecutive=2,
                                  objective_rtol=1., solution_rtol=10.)
    result, record = stop.minimize_with_stability(objective, np.ones(30),
        initial_objective=objective(np.ones(30))[0], sample_fields=sample, policy=policy,
        options=dict(maxiter=100, ftol=0., gtol=0.))
    assert record["stability_stop"] and result.status == 99 and not result.success
    assert result.nit == record["first_stable_check"]["iteration"] == 3
    assert record["objective_history"][-1] == pytest.approx(objective(result.x)[0])


def test_monitor_only_preserves_the_native_trajectory():
    diagonal = np.geomspace(1., 100., 12)
    fun = lambda x: (.5*np.dot(x, diagonal*x), diagonal*x)
    opts = dict(maxiter=80, ftol=1e-14, gtol=1e-9)
    native, _ = stop.minimize_with_stability(fun, np.ones(12), initial_objective=fun(np.ones(12))[0],
                                            options=opts)
    monitored, record = stop.minimize_with_stability(fun, np.ones(12),
        initial_objective=fun(np.ones(12))[0], sample_fields=lambda x: fields(x, x, x),
        policy=stop.StabilityPolicy(window=2, check_every=1, consecutive=1),
        stop_when_stable=False, options=opts)
    np.testing.assert_array_equal(native.x, monitored.x)
    assert native.nit == monitored.nit and native.nfev == monitored.nfev
    assert native.status == monitored.status and not record["stability_stop"]
