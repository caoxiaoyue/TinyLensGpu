"""TDD tests for PixelizedSourceModel."""

import pytest
import jax.numpy as jnp

from TinyLensGpu.Inference import ParamU
from TinyLensGpu.PhysicalModel import PhysicalModel, SIE, Shear


def _build_affine_source_grid(n: int = 5, half_size: float = 2.0):
    grid = jnp.linspace(-half_size, half_size, n)
    x_grid, y_grid = jnp.meshgrid(grid, grid, indexing="xy")
    source_values = 2.0 * x_grid - 0.5 * y_grid + 1.25
    return grid, x_grid, y_grid, source_values


@pytest.mark.unit
def test_public_import_paths_work():
    from TinyLensGpu.PhysicalModel import PixelizedSourceModel as RootExport
    from TinyLensGpu.PhysicalModel.LensImage import PixelizedSourceModel as LensImageExport

    assert RootExport is LensImageExport


@pytest.mark.unit
def test_construction_wraps_log_lambda_reg_as_paramu():
    from TinyLensGpu.PhysicalModel import PixelizedSourceModel

    model = PixelizedSourceModel(nx=40, ny=40, log_lambda_reg=jnp.log(10.0))

    assert model.nx == 40
    assert model.ny == 40
    assert isinstance(model.log_lambda_reg, ParamU)
    assert model.log_lambda_reg.prior_type == "uniform"
    assert list(model.log_lambda_reg.prior_settings) == [jnp.log(1e-4), jnp.log(1e4)]
    assert model.adaptive_reg_alpha == 0.0
    assert model.adaptive_reg_floor == 0.1


@pytest.mark.unit
def test_adaptive_hyperparams_can_be_paramu_and_dynamic():
    from TinyLensGpu.Inference.build_prior import extract_prior_specs
    from TinyLensGpu.PhysicalModel import PixelizedSourceModel

    log_lambda = ParamU(
        "log_lambda_reg", 0.0,
        prior_type="uniform", prior_settings=[-2.0, 2.0],
        limits=[-2.0, 2.0],
    )
    alpha = ParamU(
        "adaptive_reg_alpha", 1.0,
        prior_type="uniform", prior_settings=[0.0, 5.0],
        limits=[0.0, 5.0],
    )
    floor = ParamU(
        "adaptive_reg_floor", 0.1,
        prior_type="log_uniform", prior_settings=[1.0e-3, 1.0],
        limits=[1.0e-3, 1.0],
    )
    for p in (log_lambda, alpha, floor):
        p.to_dynamic()

    source = PixelizedSourceModel(
        nx=5,
        ny=5,
        log_lambda_reg=log_lambda,
        adaptive_reg_alpha=alpha,
        adaptive_reg_floor=floor,
    )

    assert source.adaptive_reg_alpha is alpha
    assert source.adaptive_reg_floor is floor
    names = [spec.name for spec in extract_prior_specs(source)]
    assert names == ["log_lambda_reg", "adaptive_reg_alpha", "adaptive_reg_floor"]


@pytest.mark.unit
def test_scalar_adaptive_hyperparams_are_not_dynamic_params():
    from TinyLensGpu.PhysicalModel import PixelizedSourceModel

    source = PixelizedSourceModel(
        nx=5,
        ny=5,
        log_lambda_reg=0.0,
        adaptive_reg_alpha=1.5,
        adaptive_reg_floor=0.2,
    )
    source.log_lambda_reg.to_dynamic()

    dynamic_names = {param.name for param in source.dynamic_params}

    assert dynamic_names == {"log_lambda_reg"}
    assert source.adaptive_reg_alpha == 1.5
    assert source.adaptive_reg_floor == 0.2


@pytest.mark.unit
def test_kernel_scale_exists_only_for_gp_regularization():
    from TinyLensGpu.PhysicalModel import PixelizedSourceModel

    gp_model = PixelizedSourceModel(
        nx=5,
        ny=5,
        log_lambda_reg=jnp.log(10.0),
        regularization_type="gaussian",
        kernel_scale=2.0,
    )
    non_gp_model = PixelizedSourceModel(
        nx=5,
        ny=5,
        log_lambda_reg=jnp.log(10.0),
        regularization_type="second-order",
    )

    assert isinstance(gp_model.kernel_scale, ParamU)
    assert not hasattr(non_gp_model, "kernel_scale") or non_gp_model.kernel_scale is None


@pytest.mark.unit
def test_pixel_brightness_values_are_not_registered_as_caskade_params():
    from TinyLensGpu.PhysicalModel import PixelizedSourceModel

    sie = SIE(theta_E=1.0, e1=0.0, e2=0.0, center_x=0.0, center_y=0.0)
    shear = Shear(gamma1=0.03, gamma2=-0.01)
    source = PixelizedSourceModel(nx=5, ny=5, log_lambda_reg=jnp.log(10.0))

    sie.theta_E.to_dynamic()
    sie.e1.to_static()
    sie.e2.to_static()
    sie.center_x.to_static()
    sie.center_y.to_static()
    shear.gamma1.to_static()
    shear.gamma2.to_static()
    source.log_lambda_reg.to_dynamic()

    model = PhysicalModel(lens_mass=[sie, shear], source_light=[source], lens_light=[])

    dynamic_names = {param.name for param in model.get_dynamic_params()}

    assert dynamic_names == {"theta_E", "log_lambda_reg"}
    assert "source_values" not in dynamic_names
    assert "brightness" not in dynamic_names


@pytest.mark.unit
def test_dynamic_params_include_kernel_scale_for_gp_regularization():
    from TinyLensGpu.PhysicalModel import PixelizedSourceModel

    sie = SIE(theta_E=1.0, e1=0.0, e2=0.0, center_x=0.0, center_y=0.0)
    source = PixelizedSourceModel(
        nx=5,
        ny=5,
        log_lambda_reg=jnp.log(10.0),
        regularization_type="gaussian",
        kernel_scale=2.0,
    )

    sie.theta_E.to_dynamic()
    sie.e1.to_static()
    sie.e2.to_static()
    sie.center_x.to_static()
    sie.center_y.to_static()
    source.log_lambda_reg.to_dynamic()
    source.kernel_scale.to_dynamic()

    model = PhysicalModel(lens_mass=[sie], source_light=[source], lens_light=[])

    dynamic_names = {param.name for param in model.get_dynamic_params()}

    assert dynamic_names == {"theta_E", "log_lambda_reg", "kernel_scale"}


@pytest.mark.unit
def test_light_returns_bilinear_interpolation_matching_input_shape():
    from TinyLensGpu.PhysicalModel import PixelizedSourceModel

    _, _, _, source_values = _build_affine_source_grid()
    model = PixelizedSourceModel(nx=5, ny=5, log_lambda_reg=jnp.log(10.0))

    x = jnp.array([[-1.5, -0.25, 0.75], [1.25, -0.5, 1.5]], dtype=jnp.float32)
    y = jnp.array([[0.25, -1.0, 1.0], [1.5, -1.25, 0.0]], dtype=jnp.float32)
    expected = 2.0 * x - 0.5 * y + 1.25

    brightness = model.light(x, y, source_values, (-2.0, 2.0, -2.0, 2.0))

    assert brightness.shape == x.shape
    assert jnp.allclose(brightness, expected)


@pytest.mark.unit
def test_light_exact_grid_points_return_input_values():
    from TinyLensGpu.PhysicalModel import PixelizedSourceModel

    _, x_grid, y_grid, source_values = _build_affine_source_grid()
    model = PixelizedSourceModel(nx=5, ny=5, log_lambda_reg=jnp.log(10.0))

    brightness = model.light(x_grid, y_grid, source_values, (-2.0, 2.0, -2.0, 2.0))

    assert brightness.shape == source_values.shape
    assert jnp.allclose(brightness, source_values)


@pytest.mark.unit
def test_invalid_regularization_type_raises_value_error():
    from TinyLensGpu.PhysicalModel import PixelizedSourceModel

    with pytest.raises(ValueError):
        PixelizedSourceModel(nx=5, ny=5, log_lambda_reg=jnp.log(10.0), regularization_type="invalid")


if __name__ == "__main__":
    pytest.main()
