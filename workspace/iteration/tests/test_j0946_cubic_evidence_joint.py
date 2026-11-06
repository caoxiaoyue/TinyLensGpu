"""Changing lambda_s must update objective, whitening and normalized evidence."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from test_j0946_cubic_source import load, problems


@pytest.mark.parametrize("lambda_s", [.0003, .2, 17.])
def test_source_strength_updates_entire_joint_prior_and_evidence(lambda_s):
    module = load("lbfgs.py", "test_cubic_joint_source_lambda")
    with jax.experimental.enable_x64(True):
        hard, smooth = problems()
        reference = module.EvidenceEngine(hard, maxiter=2)
        engine = module.EvidenceEngine(smooth, maxiter=2, reference_engine=reference)
        logs = np.log10([lambda_s, 3., 5.])
        strengths = jnp.asarray(10.**logs)
        row, q = engine.fit(logs, initial_values=np.asarray(smooth.initial_values), audit=True)
        prior = np.asarray(smooth.dense_prior(float(strengths[0]), float(strengths[1])))
        jac = np.asarray(jax.jacfwd(smooth.forward)(jnp.asarray(q)))/np.asarray(smooth.noise_active)[:, None]
        hessian = prior+jac.T@jac
        logdet_prior = np.linalg.slogdet(prior)[1]
        logdet_ratio = np.linalg.slogdet(hessian)[1]-logdet_prior
        expected = -row["objective"]-.5*(logdet_ratio+engine.noise_normalization)
        np.testing.assert_allclose(row["logdet_prior"], logdet_prior, atol=1e-9)
        np.testing.assert_allclose(row["diagnostic_log_evidence"], expected, atol=1e-8)
        value, gradient = engine.physical_value_grad(jnp.asarray(q), strengths)
        point = smooth._evaluate_kernel(jnp.asarray(q), float(strengths[0]), float(strengths[1]))
        np.testing.assert_allclose(value, point.objective, atol=1e-10)
        np.testing.assert_allclose(gradient, -point.negative_gradient, atol=1e-8)
        proxy = .5*np.asarray(gradient)@np.linalg.solve(hessian, np.asarray(gradient))
        np.testing.assert_allclose(row["gn_stationarity_proxy"], proxy, rtol=1e-10, atol=1e-8)
        assert row["algebra_audit"]["whitening_pass"]
        assert row["algebra_audit"]["determinant_pass"]
