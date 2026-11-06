"""Matched coarse-noise generation and actual lens likelihood weighting."""

import importlib.util
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from TinyLensGpu.ForwardSimulation import PotentialCorrectionRenderer,SimulatorConfig
from TinyLensGpu.PhysicalModel import PhysicalModel,Shear


HERE=Path(__file__).resolve().parents[1]


def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result)
    return result


noise=module("coarse_noise_test",HERE/"coarse_noise.py")
observation=module("noisy_projection_test",HERE/"data_resolution_observation.py")


def test_extra_noise_variance_and_reproducible_shared_data():
    data=np.linspace(0,1,2312)
    old=np.full(data.size,.01/3)
    first=noise.add_matched_noise(data,old,.01,0)
    again=noise.add_matched_noise(data,old,.01,0)
    other=noise.add_matched_noise(data,old,.01,1)
    np.testing.assert_array_equal(first["data"],again["data"])
    assert not np.array_equal(first["data"],other["data"])
    np.testing.assert_allclose(first["extra_sigma"],np.sqrt(.01**2-(.01/3)**2),atol=1e-16)
    np.testing.assert_allclose(first["base_sigma"]**2+first["extra_sigma"]**2,first["sigma"]**2,atol=1e-20)
    np.testing.assert_allclose(first["data"]-data,first["extra_noise"],atol=1e-16)


def test_cannot_reduce_variance_and_equal_sigma_preserves_data():
    data=np.arange(8.)
    with pytest.raises(ValueError,match="cannot lower"):
        noise.add_matched_noise(data,.01,.003,0)
    same=noise.add_matched_noise(data,.01,.01,0)
    np.testing.assert_array_equal(same["data"],data)
    np.testing.assert_array_equal(same["extra_noise"],np.zeros_like(data))


def test_actual_projected_lens_uses_replaced_data_noise_and_normalization():
    with jax.experimental.enable_x64(True):
        mask=np.zeros((9,9),bool)
        model=PhysicalModel(lens_mass=[Shear(gamma1=.02,gamma2=.01)],source_light=[],lens_light=[])
        renderer=PotentialCorrectionRenderer(model,SimulatorConfig(.1,9,jnp.ones((1,1)),mask=mask),
            4,(-.7,.7,-.7,.7),potential_spacing=.2,dtype="float64")
        projection=observation.ObservationProjection(mask,3)
        s=jnp.asarray(np.random.default_rng(9).normal(size=16))
        phi=jnp.asarray(np.random.default_rng(1).normal(scale=.0001,size=renderer.potential_grid.n_unknown))
        q=jnp.r_[s,phi]
        fine=renderer.render(s,phi).model_image
        problem=observation.ProjectedPotentialCorrection(renderer,fine,jnp.full((9,9),.01),s,
            projection=projection,lambda_amplitude=.3,source_regularization="second-order")
        changed=noise.add_matched_noise(problem.data_active,problem.noise_active,.01,2)
        noise.set_projected_likelihood(problem,changed["data"],changed["sigma"])
        np.testing.assert_allclose(problem.forward(q),projection.apply(fine.ravel()),atol=1e-13)
        np.testing.assert_array_equal(problem.noise_active,np.full(projection.n_data,.01))
        np.testing.assert_allclose(problem.noise_logdet,2*projection.n_data*np.log(.01),atol=1e-13)
        def objective(point):
            residual=(jnp.asarray(changed["data"])-problem.forward(point))/.01
            return .5*(residual@residual+point@problem.prior_matvec(point,.02,.1))
        value,gradient=jax.value_and_grad(objective)(q)
        actual=problem._evaluate_kernel(q,.02,.1)
        np.testing.assert_allclose(actual.objective,value,rtol=1e-12)
        np.testing.assert_allclose(-actual.negative_gradient,gradient,rtol=1e-11,atol=1e-9)
        np.testing.assert_allclose(problem._result_products(q,.02,.1)["objective"],value,rtol=1e-12)
        assert problem.renderer.potential_grid is renderer.potential_grid
