"""Exact rebinning, covariance and lens derivatives for the data-size probe."""

import importlib.util
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from TinyLensGpu.ForwardSimulation import PotentialCorrectionRenderer,SimulatorConfig
from TinyLensGpu.PhysicalModel import PhysicalModel,Shear


PATH=Path(__file__).resolve().parents[1]/"data_resolution_observation.py"
spec=importlib.util.spec_from_file_location("data_projection_test",PATH)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
Projection=module.ObservationProjection


def test_disjoint_block_mean_covariance_and_adjoint():
    with jax.experimental.enable_x64(True):
        mask=np.zeros((12,12),bool);mask[0,0]=True
        p=Projection(mask,3)
        rng=np.random.default_rng(3)
        x=jnp.asarray(rng.normal(size=p.n_fine));u=jnp.asarray(rng.normal(size=p.n_data))
        sigma=jnp.asarray(np.linspace(.1,.3,p.n_fine))
        b=np.zeros((p.n_data,p.n_fine))
        for i,idx in enumerate(np.asarray(p.indices)): b[i,idx]=1/9
        np.testing.assert_allclose(p.apply(x),b@x,atol=1e-15)
        np.testing.assert_allclose(jnp.vdot(u,p.apply(x)),jnp.vdot(p.adjoint(u),x),atol=1e-15)
        np.testing.assert_allclose(np.diag(np.asarray(p.noise(sigma))**2),b@np.diag(np.asarray(sigma)**2)@b.T,atol=1e-16)
        np.testing.assert_allclose(p.apply(jnp.ones(p.n_fine)),1)
        np.testing.assert_allclose(p.noise(jnp.full(p.n_fine,.01)),.01/3)
        assert p.n_data==15


def test_matched_footprint_keeps_exactly_the_used_fine_pixels():
    with jax.experimental.enable_x64(True):
        mask=np.zeros((14,14),bool);mask[2,2]=True
        coarse=Projection(mask,3)
        matched=Projection(mask,matched_to=coarse)
        assert matched.n_data==9*coarse.n_data
        np.testing.assert_array_equal(np.asarray(matched.indices).ravel(),np.unique(np.asarray(coarse.indices)))
        fine=jnp.arange(coarse.n_fine,dtype=jnp.float64)
        np.testing.assert_array_equal(matched.apply(fine),fine[np.asarray(matched.indices).ravel()])


def test_projected_lens_forward_analytic_derivatives_and_full_objective():
    with jax.experimental.enable_x64(True):
        mask=np.zeros((12,12),bool)
        psf=jnp.asarray([[1.,2,1],[2,4,2],[1,2,1]])/16
        model=PhysicalModel(lens_mass=[Shear(gamma1=.02,gamma2=.01)],source_light=[],lens_light=[])
        renderer=PotentialCorrectionRenderer(model,SimulatorConfig(.1,12,psf,nsub=2,mask=mask),
            5,(-.7,.7,-.7,.7),potential_spacing=.2,dtype="float64")
        p=Projection(mask,3)
        rng=np.random.default_rng(12)
        source=jnp.asarray(rng.normal(size=25))
        q=jnp.r_[source,jnp.asarray(rng.normal(scale=.0001,size=renderer.potential_grid.n_unknown))]
        data=renderer.render(q[:25],q[25:]).model_image+.01
        problem=module.ProjectedPotentialCorrection(renderer,data,jnp.full((12,12),.02),source,
            projection=p,lambda_amplitude=.3,source_regularization="second-order")
        fine=renderer.render(q[:25],q[25:]).image_active
        np.testing.assert_allclose(problem.forward(q),p.apply(fine),atol=1e-13)
        np.testing.assert_allclose(problem.data_active,p.apply(data.ravel()),atol=1e-13)
        np.testing.assert_allclose(problem.noise_active,.02/3,atol=1e-15)
        assert problem.renderer.potential_grid is renderer.potential_grid
        lin=problem.linearize(q)
        direction=jnp.asarray(rng.normal(size=q.size));u=jnp.asarray(rng.normal(size=p.n_data))
        jv=jax.jvp(problem.forward,(q,),(direction,))[1]
        np.testing.assert_allclose(problem.jacobian_forward(direction,lin),jv,rtol=1e-11,atol=1e-11)
        np.testing.assert_allclose(jnp.vdot(u,jv),jnp.vdot(problem.jacobian_adjoint(u,lin),direction),rtol=1e-11,atol=1e-11)
        def objective(point):
            residual=(problem.data_active-problem.forward(point))/problem.noise_active
            return .5*(residual@residual+point@problem.prior_matvec(point,.02,.1))
        f,g=jax.value_and_grad(objective)(q)
        analytic=problem._evaluate_kernel(q,.02,.1)
        np.testing.assert_allclose(f,analytic.objective,rtol=1e-12)
        np.testing.assert_allclose(g,-analytic.negative_gradient,rtol=1e-11,atol=1e-8)
        products=problem._result_products(q,.02,.1)
        np.testing.assert_allclose(products["objective"],f,rtol=1e-12)
        np.testing.assert_allclose(products["nonlinear_image_active"],problem.forward(q),atol=1e-13)


def test_binning_cannot_increase_information_for_fixed_fine_signal():
    with jax.experimental.enable_x64(True):
        p=Projection(np.zeros((12,12),bool),3)
        rng=np.random.default_rng(19)
        for _ in range(3):
            signal=jnp.asarray(rng.normal(size=p.n_fine));sigma=jnp.asarray(rng.uniform(.1,.3,size=p.n_fine))
            coarse=float(jnp.sum((p.apply(signal)/p.noise(sigma))**2))
            fine=float(jnp.sum((signal/sigma)**2))
            assert coarse<=fine+1e-10
