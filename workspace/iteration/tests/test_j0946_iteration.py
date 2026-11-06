"""Algebra and handoff regressions for the workspace-only J0946 experiment."""

import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[3]
ITERATION = ROOT / "workspace" / "iteration"


def module(name, filename):
    spec = importlib.util.spec_from_file_location(name, ITERATION / filename)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def test_full_coupling_whitening_gradient_determinant_and_stationarity():
    lbfgs = module("iteration_lbfgs_test", "lbfgs.py")
    with jax.experimental.enable_x64(True):
        rng = np.random.default_rng(94)
        ns, npot, nd = 7, 4, 5
        root = rng.normal(size=(ns, ns))
        rs = root @ root.T + np.diag(np.linspace(1, 50, ns))
        difference = rng.normal(size=(6, npot))
        r4 = difference.T @ difference
        jac = rng.normal(size=(nd, ns + npot))
        sd, sc, _ = lbfgs.scaled_cholesky(jnp.asarray(rs))
        s0 = lbfgs.prior_response(jnp.asarray(jac[:, :ns].T), sd, sc)
        # Changing lambda0 must change prior, transform and normalized evidence.
        logdets = []
        for l0 in (0.03, 500):
            ls, l4 = 0.2, 300
            qp = l4 * r4 + l0 * np.eye(npot)
            prior = np.zeros((ns + npot, ns + npot))
            prior[:ns, :ns] = ls * rs
            prior[ns:, ns:] = qp
            curvature = prior + jac.T @ jac
            pd, pc, _ = lbfgs.scaled_cholesky(jnp.asarray(qp))
            vp = lbfgs.prior_response(jnp.asarray(jac[:, ns:].T), pd, pc)
            vs = s0 / np.sqrt(ls)
            covariance = jnp.eye(nd) + vs.T @ vs + vp.T @ vp
            cd, cc, logdet = lbfgs.scaled_cholesky(covariance)
            eigenvalues, vectors = jnp.linalg.eigh(covariance)
            roots = jnp.sqrt(eigenvalues)
            coefficients = 1 / (roots * (1 + roots))

            def transform(z):
                return lbfgs.low_rank_whiten(z, sd, sc, s0, pd, pc, vp,
                                             ls, vectors, coefficients)

            transform_matrix = np.asarray(jax.jacfwd(transform)(jnp.zeros(ns + npot)))
            np.testing.assert_allclose(transform_matrix.T @ curvature @ transform_matrix,
                                       np.eye(ns + npot), atol=1e-11)
            expected_ld = np.linalg.slogdet(curvature)[1] - np.linalg.slogdet(prior)[1]
            np.testing.assert_allclose(logdet, expected_ld, atol=1e-11)
            gradient = rng.normal(size=ns + npot)
            chain = jax.grad(lambda z: jnp.vdot(transform(z), gradient))(jnp.ones(ns + npot))
            np.testing.assert_allclose(chain, transform_matrix.T @ gradient, atol=1e-12)
            source_root = np.diag(np.asarray(sd)) @ np.asarray(sc) * np.sqrt(ls)
            potential_root = np.diag(np.asarray(pd)) @ np.asarray(pc)
            white_gradient = np.r_[np.linalg.solve(source_root, gradient[:ns]),
                                   np.linalg.solve(potential_root, gradient[ns:])]
            proxy = lbfgs.stationarity_proxy(vs, vp, jnp.asarray(white_gradient), cd, cc)
            np.testing.assert_allclose(proxy, 0.5 * gradient @ np.linalg.solve(curvature, gradient),
                                       atol=1e-12)
            logdets.append(float(logdet))
        assert logdets[0] != logdets[1]


def test_handoff_missing_and_checksum_fail_without_running_smooth(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(ITERATION))
    pt = module("iteration_pt_test", "pt.py")
    with pytest.raises(FileNotFoundError, match="never run automatically"):
        pt.load_handoff(tmp_path)
    metadata = dict(complete=True, handoff_sha256="incorrect", source_n=2)
    (tmp_path / "m3_handoff.json").write_text(json.dumps(metadata))
    np.savez(tmp_path / "m3_handoff.npz", source_scale=np.ones(4))
    with pytest.raises(ValueError, match="checksum"):
        pt.load_handoff(tmp_path)
    metadata["handoff_sha256"] = hashlib.sha256((tmp_path / "m3_handoff.npz").read_bytes()).hexdigest()
    (tmp_path / "m3_handoff.json").write_text(json.dumps(metadata))
    _, arrays = pt.load_handoff(tmp_path)
    np.testing.assert_array_equal(arrays["source_scale"], np.ones(4))
    assert not (tmp_path / "macro_model.py").exists()


def test_conditional_source_start_matches_direct_adaptive_normal_equations():
    lbfgs = module("iteration_conditional_test", "lbfgs.py")
    with jax.experimental.enable_x64(True):
        rng = np.random.default_rng(946)
        ns, nd = 6, 9
        root = rng.normal(size=(ns, ns))
        precision = root @ root.T + np.diag(np.arange(1, ns+1)**2)
        design = rng.normal(size=(nd, ns))
        data = rng.normal(size=nd)
        noise = np.linspace(0.2, 3, nd)
        weighted = design / noise[:, None]
        engine = lbfgs.EvidenceEngine.__new__(lbfgs.EvidenceEngine)
        engine.ns, engine.np = ns, 3
        engine.eye = jnp.eye(nd)
        engine.sd, engine.sc, _ = lbfgs.scaled_cholesky(jnp.asarray(precision))
        engine.s0 = lbfgs.prior_response(jnp.asarray(weighted.T), engine.sd, engine.sc)
        engine.source_cov0 = engine.s0.T @ engine.s0
        engine.problem = SimpleNamespace(data_active=jnp.asarray(data), noise_active=jnp.asarray(noise))
        for strength in (0.003, 1.0, 1000.0):
            initial = np.asarray(engine.conditional_source(strength))
            expected = np.linalg.solve(weighted.T @ weighted + strength * precision,
                                       weighted.T @ (data/noise))
            np.testing.assert_allclose(initial[:ns], expected, rtol=1e-11, atol=1e-11)
            np.testing.assert_array_equal(initial[ns:], np.zeros(3))


def test_saved_scale_is_required_positive(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(ITERATION))
    pt = module("iteration_pt_scale_test", "pt.py")
    np.savez(tmp_path / "m3_handoff.npz", source_scale=np.array([1, 2, 0, 4.]))
    metadata = dict(complete=True, source_n=2,
        handoff_sha256=hashlib.sha256((tmp_path / "m3_handoff.npz").read_bytes()).hexdigest())
    (tmp_path / "m3_handoff.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="positive"):
        pt.load_handoff(tmp_path)
