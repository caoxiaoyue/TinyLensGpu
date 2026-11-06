"""C1 source-node semantics, derivatives, operator transpose and GN evidence."""

import importlib.util
from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np
from scipy.interpolate import CubicHermiteSpline

from TinyLensGpu.ForwardSimulation import PotentialCorrectionRenderer, SimulatorConfig
from TinyLensGpu.ObservationModel import IterativePotentialCorrection
from TinyLensGpu.PhysicalModel import PhysicalModel, Shear


HERE = Path(__file__).resolve().parents[1]


def load(filename, name):
    spec = importlib.util.spec_from_file_location(name, HERE/filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


cubic = load("source_cubic.py", "test_cubic_source_basis")


def brightness(source, points, bbox, n):
    w, idx, _ = cubic.cubic_mapping(points, bbox, n)
    return jnp.sum(w*source[idx], axis=1)


def test_original_node_values_and_independent_tensor_hermite_reference():
    with jax.experimental.enable_x64(True):
        n = 7
        rng = np.random.default_rng(946)
        values = rng.normal(size=(n, n))
        bbox = (-.3, .3, -.3, .3)
        axis = np.linspace(-.3, .3, n)
        x, y = np.meshgrid(axis, axis)
        nodes = jnp.asarray(np.column_stack((x.ravel(), y.ravel())))
        np.testing.assert_allclose(brightness(jnp.asarray(values.ravel()), nodes, bbox, n), values.ravel(), atol=1e-14)
        # Independent SciPy Hermite polynomials, including the compact zero tails.
        padded = np.pad(values, 2)
        extended_axis = np.arange(-2, n+2)*.1-.3
        slopes_x = np.zeros_like(padded)
        slopes_x[:, 1:-1] = (padded[:, 2:]-padded[:, :-2])/.2
        points = rng.uniform(-.52, .52, (40, 2))
        expected, gradient = [], []
        for px, py in points:
            if abs(px) > .5 or abs(py) > .5:
                expected.append(0.); gradient.append([0., 0.])
                continue
            along_x = CubicHermiteSpline(extended_axis, padded, slopes_x, axis=1, extrapolate=False)
            vx, dx = along_x(px), along_x(px, 1)
            sy = np.r_[0., (vx[2:]-vx[:-2])/.2, 0.]
            dsy = np.r_[0., (dx[2:]-dx[:-2])/.2, 0.]
            spline_y = CubicHermiteSpline(extended_axis, vx, sy, extrapolate=False)
            spline_dx = CubicHermiteSpline(extended_axis, dx, dsy, extrapolate=False)
            expected.append(spline_y(py))
            gradient.append([spline_dx(py), spline_y(py, 1)])
        actual = brightness(jnp.asarray(values.ravel()), jnp.asarray(points), bbox, n)
        gx, gy = cubic.cubic_gradient(jnp.asarray(values.ravel()), jnp.asarray(points), bbox, n)
        np.testing.assert_allclose(actual, expected, atol=2e-14)
        np.testing.assert_allclose(np.column_stack((gx, gy)), gradient, atol=2e-13)


def test_c1_at_interior_edge_ghost_and_support_gridlines():
    with jax.experimental.enable_x64(True):
        source = jnp.asarray(np.random.default_rng(31).normal(size=49))
        n, bbox = 7, (-.3, .3, -.3, .3)
        # Deliberately cover every original node, both ghost lines, and zero support.
        lines = np.arange(-2, n+2)*.1-.3
        centers = np.r_[np.column_stack((lines, np.full(len(lines), .037))),
                        np.column_stack((np.full(len(lines), -.062), lines))]
        direction = np.r_[np.tile([1., 0.], (len(lines), 1)), np.tile([0., 1.], (len(lines), 1))]
        jumps = []
        for h in (1e-6, 1e-9):
            minus, plus = jnp.asarray(centers-h*direction), jnp.asarray(centers+h*direction)
            gm = np.column_stack(cubic.cubic_gradient(source, minus, bbox, n))
            gp = np.column_stack(cubic.cubic_gradient(source, plus, bbox, n))
            jumps.append(np.max(abs(gm-gp)))
        assert jumps[-1] < 1e-4
        assert jumps[-1] < jumps[0]/500
        ad = jax.jacfwd(lambda p: brightness(source, p[None, :], bbox, n)[0])
        analytic = np.column_stack(cubic.cubic_gradient(source, jnp.asarray(centers), bbox, n))
        np.testing.assert_allclose(jax.vmap(ad)(jnp.asarray(centers)), analytic, atol=3e-13)
        # An affine brightness field is exactly reproduced away from zero tails.
        axis = np.linspace(-.3, .3, n)
        x, y = np.meshgrid(axis, axis)
        plane = jnp.asarray((2+3*x-4*y).ravel())
        points = jnp.array([[.031, -.084], [-.145, .074]])
        np.testing.assert_allclose(brightness(plane, points, bbox, n), 2+3*points[:, 0]-4*points[:, 1], atol=2e-14)


def problems():
    psf = jnp.array([[0., .1, 0.], [.1, .6, .1], [0., .1, 0.]])
    config = SimulatorConfig(.06, 3, psf, nsub=2, mask=np.zeros((3, 3), bool))
    macro = PhysicalModel(lens_mass=[Shear(gamma1=0., gamma2=0.)], source_light=[], lens_light=[])
    hard = PotentialCorrectionRenderer(macro, config, 5, (-.1, .1, -.1, .1), dtype="float64")
    smooth = cubic.CubicSourceRenderer(macro, config, 5, hard.source_bbox, potential_grid=hard.potential_grid, dtype="float64")
    result = []
    for renderer in (hard, smooth):
        p = IterativePotentialCorrection(renderer, np.arange(9).reshape(3, 3)*.005,
            np.full((3, 3), .01), np.linspace(.01, .08, 25),
            lambda_amplitude=5., source_regularization="first-order")
        p.source_scale = jnp.linspace(1., 2., 25)
        result.append(p)
    return result


def test_forward_ad_gradient_and_transpose_match_analytic_operators():
    with jax.experimental.enable_x64(True):
        _, problem = problems()
        rng = np.random.default_rng(51)
        q = problem.initial_values+jnp.asarray(rng.normal(size=problem.n_joint))*1e-6
        point = problem._evaluate_kernel(q, .2, 3.)
        def objective(x):
            residual = (problem.forward(x)-problem.data_active)/problem.noise_active
            return .5*(jnp.vdot(residual, residual)+jnp.vdot(x, problem.prior_matvec(x, .2, 3.)))
        np.testing.assert_allclose(jax.grad(objective)(q), -point.negative_gradient, rtol=2e-12, atol=1e-9)
        linearization = problem.linearize(q)
        image = jnp.asarray(rng.normal(size=problem.renderer.n_active))
        delta = jnp.asarray(rng.normal(size=problem.n_joint))
        np.testing.assert_allclose(jnp.vdot(problem.jacobian_forward(delta, linearization), image),
                                   jnp.vdot(delta, problem.jacobian_adjoint(image, linearization)), atol=2e-11)
        source = delta[:problem.n_source]
        np.testing.assert_allclose(jnp.vdot(problem.renderer.source_forward(source, linearization.mapping), image),
                                   jnp.vdot(source, problem.renderer.source_adjoint(image, linearization.mapping)), atol=2e-12)


def test_same_prior_shared_whitening_and_full_coupled_evidence():
    engine_module = load("lbfgs.py", "test_cubic_source_evidence")
    with jax.experimental.enable_x64(True):
        hard, smooth = problems()
        np.testing.assert_array_equal(hard.dense_prior(.2, 3.), smooth.dense_prior(.2, 3.))
        reference = engine_module.EvidenceEngine(hard, maxiter=2)
        engine = engine_module.EvidenceEngine(smooth, maxiter=2, reference_engine=reference)
        logs = np.log10([.2, 3., 5.]); strengths = 10.**logs
        initial = smooth.initial_values
        row, q = engine.fit(logs, initial_values=initial, audit=True)
        assert row["algebra_audit"]["whitening_pass"] and row["algebra_audit"]["determinant_pass"]
        jac = np.asarray(jax.jacfwd(smooth.forward)(jnp.asarray(q)))/np.asarray(smooth.noise_active)[:, None]
        prior = np.asarray(smooth.dense_prior(strengths[0], strengths[1]))
        hessian = prior+jac.T@jac
        ratio = np.linalg.slogdet(hessian)[1]-np.linalg.slogdet(prior)[1]
        expected = -row["objective"]-.5*(ratio+engine.noise_normalization)
        np.testing.assert_allclose(row["diagnostic_log_evidence"], expected, atol=1e-8)
        _, grad = engine.physical_value_grad(jnp.asarray(q), jnp.asarray(strengths))
        np.testing.assert_allclose(row["gn_stationarity_proxy"], .5*np.asarray(grad)@np.linalg.solve(hessian, np.asarray(grad)), atol=1e-9)
