"""Public supplied-field rendering and physical potential operator contracts."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from TinyLensGpu.utils.lensing.potential import PotentialGrid


def test_isolated_query_constructs_local_support_without_rectangular_padding():
    with jax.experimental.enable_x64():
        grid = PotentialGrid.from_queries(
            jnp.array([0.25]),
            jnp.array([0.75]),
            spacing=1.0,
            dtype="float64",
        )

    np.testing.assert_array_equal(
        np.asarray(grid.node_indices)[np.asarray(grid.core_indices)],
        [[0, 0], [1, 0], [0, 1], [1, 1]],
    )
    assert grid.n_core == 4
    assert grid.n_unknown == 20
    assert np.unique(np.asarray(grid.node_indices), axis=0).shape[0] == 20
    assert (-2, 0) in map(tuple, np.asarray(grid.node_indices))
    assert (-2, -2) not in map(tuple, np.asarray(grid.node_indices))
    assert np.asarray(grid.derivative_valid).sum() == 4
    np.testing.assert_allclose(
        grid.interpolation_weights, [[0.1875, 0.0625, 0.5625, 0.1875]]
    )


def test_quadratic_potential_has_physical_derivatives_and_interpolated_deflection():
    with jax.experimental.enable_x64():
        qx = jnp.array([-0.17, 0.0, 0.39])
        qy = jnp.array([0.31, -0.4, 0.05])
        grid = PotentialGrid.from_queries(qx, qy, spacing=0.2)
        p = 2 * grid.x**2 + 3 * grid.y**2 + 0.5 * grid.x * grid.y + 0.7 * grid.x
        derivatives = grid.derivatives(p)
        cx, cy = np.asarray(grid.x[grid.core_indices]), np.asarray(
            grid.y[grid.core_indices]
        )
        alpha_x, alpha_y = jax.jit(grid.deflection)(p)

    qx, qy = np.asarray(qx), np.asarray(qy)
    np.testing.assert_allclose(
        derivatives.deflection_x, 4 * cx + 0.5 * cy + 0.7, atol=2e-14
    )
    np.testing.assert_allclose(derivatives.deflection_y, 6 * cy + 0.5 * cx, atol=2e-14)
    np.testing.assert_allclose(derivatives.convergence, 5.0, atol=3e-14)
    np.testing.assert_allclose(alpha_x, 4 * qx + 0.5 * qy + 0.7, atol=2e-14)
    np.testing.assert_allclose(alpha_y, 6 * qy + 0.5 * qx, atol=2e-14)


def test_potential_deflection_and_transpose_satisfy_inner_product_contract():
    rng = np.random.default_rng(32)
    with jax.experimental.enable_x64():
        grid = PotentialGrid.from_queries(
            rng.uniform(-0.5, 0.5, 13),
            rng.uniform(-0.5, 0.5, 13),
            spacing=0.13,
        )
        p = jnp.asarray(rng.normal(size=grid.n_unknown))
        vx, vy = jnp.asarray(rng.normal(size=(2, 13)))
        ax, ay = grid.deflection(p)
        transpose = jax.jit(grid.deflection_adjoint)(vx, vy)
        lhs, rhs = jnp.vdot(vx, ax) + jnp.vdot(vy, ay), jnp.vdot(transpose, p)
    np.testing.assert_allclose(lhs, rhs, rtol=2e-14, atol=3e-14)


@pytest.mark.parametrize("nsub", [1, 3])
def test_zero_correction_preserves_existing_pixelized_imaging(nsub):
    from TinyLensGpu.ForwardSimulation.LensImage.config import SimulatorConfig
    from TinyLensGpu.ForwardSimulation.LensImage.pixelized_operator import (
        PixelizedLensOperator,
    )
    from TinyLensGpu.ForwardSimulation.LensImage.potential_correction import (
        PotentialCorrectionRenderer,
    )
    from TinyLensGpu.PhysicalModel import PhysicalModel, Shear, PixelizedSourceModel

    with jax.experimental.enable_x64():
        mask = np.ones((7, 7), dtype=bool)
        mask[1:6, 1:6] = False
        mask[3, 3] = True
        mask[0, 0] = False
        kernel = jnp.array([[0.0, 0.1, 0.0], [0.05, 0.6, 0.1], [0.0, 0.15, 0.0]])
        config = SimulatorConfig(0.1, 7, kernel, nsub=nsub, mask=mask)
        macro = PhysicalModel(
            lens_mass=[Shear(gamma1=0.1, gamma2=0.03)], source_light=[], lens_light=[]
        )
        physical = PhysicalModel(
            lens_mass=macro.lens_mass,
            source_light=[PixelizedSourceModel(5)],
            lens_light=[],
        )
        original = PixelizedLensOperator(physical, config)
        source = jnp.asarray(np.random.default_rng(7).normal(size=25))
        bbox = (-0.6, 0.6, -0.6, 0.6)
        expected = original.forward_model(source, *bbox)
        renderer = PotentialCorrectionRenderer(macro, config, 5, bbox, dtype="float64")
        result = renderer.render(source, jnp.zeros(renderer.potential_grid.n_unknown))

    np.testing.assert_allclose(result.image_active, expected, rtol=2e-13, atol=1e-14)
    np.testing.assert_allclose(
        np.asarray(result.model_image)[~mask], expected, rtol=2e-13, atol=1e-14
    )
    assert result.image_active.shape == (np.count_nonzero(~mask),)
    assert renderer.potential_grid.spacing == pytest.approx(0.2)
    assert np.isnan(
        np.asarray(result.convergence)[~np.asarray(result.derivative_valid)]
    ).all()


@pytest.mark.parametrize("field", ["source", "subpixel_image"])
def test_corrected_source_and_imaging_actions_have_exact_transposes(field):
    from TinyLensGpu.ForwardSimulation.LensImage.config import SimulatorConfig
    from TinyLensGpu.ForwardSimulation.LensImage.potential_correction import (
        PotentialCorrectionRenderer,
    )
    from TinyLensGpu.PhysicalModel import PhysicalModel, Shear

    rng = np.random.default_rng(19)
    with jax.experimental.enable_x64():
        mask = ~np.eye(6, dtype=bool)
        mask[2, 3] = False
        kernel = jnp.array([[0.0, 0.1, 0.1], [0.05, 0.5, 0.1], [0.0, 0.05, 0.1]])
        config = SimulatorConfig(0.1, 6, kernel, nsub=2, mask=mask)
        macro = PhysicalModel(
            lens_mass=[Shear(gamma1=0.0, gamma2=0.0)], source_light=[], lens_light=[]
        )
        renderer = PotentialCorrectionRenderer(macro, config, 4, (-0.5, 0.5, -0.5, 0.5))
        p = 0.003 * renderer.potential_grid.x**2
        mapping = renderer.source_mapping(p)
        y = jnp.asarray(rng.normal(size=renderer.n_active))
        if field == "source":
            x = jnp.asarray(rng.normal(size=renderer.n_source))
            forward = jax.jit(lambda values: renderer.source_forward(values, mapping))(
                x
            )
            transpose = jax.jit(
                lambda values: renderer.source_adjoint(values, mapping)
            )(y)
        else:
            x = jnp.asarray(rng.normal(size=renderer.query_x.size))
            forward = jax.jit(renderer.image_forward)(x)
            transpose = jax.jit(renderer.image_adjoint)(y)
        lhs, rhs = jnp.vdot(y, forward), jnp.vdot(x, transpose)

    np.testing.assert_allclose(lhs, rhs, rtol=3e-14, atol=3e-14)


def test_render_subtracts_physical_correction_from_macro_source_coordinates():
    from TinyLensGpu.ForwardSimulation.LensImage.config import SimulatorConfig
    from TinyLensGpu.ForwardSimulation.LensImage.potential_correction import (
        PotentialCorrectionRenderer,
    )
    from TinyLensGpu.PhysicalModel import PhysicalModel, Shear

    with jax.experimental.enable_x64():
        mask = np.zeros((6, 6), dtype=bool)
        mask[1, 2] = True
        config = SimulatorConfig(0.1, 6, jnp.ones((1, 1)), nsub=2, mask=mask)
        macro = PhysicalModel(
            lens_mass=[Shear(gamma1=0.2, gamma2=0.1)], source_light=[], lens_light=[]
        )
        renderer = PotentialCorrectionRenderer(
            macro, config, 7, (-1.0, 1.0, -1.0, 1.0), potential_spacing=0.17
        )
        sx, sy = np.meshgrid(
            np.asarray(renderer.source_x_axis), np.asarray(renderer.source_y_axis)
        )
        source = jnp.asarray((2 * sx - 3 * sy + 5).ravel())
        px, py = renderer.potential_grid.x, renderer.potential_grid.y
        p = 0.004 * px**2 + 0.002 * px * py - 0.007 * py**2 + 0.003 * px - 0.002 * py
        result = renderer.render(source, p)

    x, y = np.asarray(renderer.query_x), np.asarray(renderer.query_y)
    bx = 0.8 * x - 0.1 * y - (0.008 * x + 0.002 * y + 0.003)
    by = 1.2 * y - 0.1 * x - (0.002 * x - 0.014 * y - 0.002)
    np.testing.assert_allclose(result.corrected_beta_x, bx, atol=2e-16)
    np.testing.assert_allclose(result.corrected_beta_y, by, atol=2e-16)
    expected_sub = np.zeros(config.subgrid_shape)
    expected_sub.ravel()[np.asarray(config.flat_indices)] = 2 * bx - 3 * by + 5
    expected = expected_sub.reshape(6, 2, 6, 2).mean(axis=(1, 3))
    np.testing.assert_allclose(result.image_active, expected[~mask], atol=2e-14)


@pytest.mark.parametrize(
    "selection", ["isolated_edge", "narrow_arc", "holes", "disconnected"]
)
def test_support_keeps_actual_subpixel_corners_for_irregular_observations(selection):
    from TinyLensGpu.ForwardSimulation.LensImage.config import SimulatorConfig
    from TinyLensGpu.ForwardSimulation.LensImage.potential_correction import (
        PotentialCorrectionRenderer,
    )
    from TinyLensGpu.PhysicalModel import PhysicalModel, Shear

    mask = np.ones((9, 9), dtype=bool)
    if selection == "isolated_edge":
        mask[0, 8] = False
    elif selection == "narrow_arc":
        mask[2:7, 6] = False
        mask[6, 3:6] = False
    elif selection == "holes":
        mask[1:8, 1:8] = False
        mask[3:6, 3:6] = True
    else:
        mask[0:2, 0:2] = False
        mask[6:8, 6:8] = False
    with jax.experimental.enable_x64():
        config = SimulatorConfig(0.1, 9, jnp.ones((1, 1)), nsub=3, mask=mask)
        macro = PhysicalModel(
            lens_mass=[Shear(gamma1=0.0, gamma2=0.0)], source_light=[], lens_light=[]
        )
        renderer = PotentialCorrectionRenderer(
            macro, config, 4, (-1.0, 1.0, -1.0, 1.0), potential_spacing=0.21
        )
        grid = renderer.potential_grid
        result = renderer.render(jnp.ones(16), jnp.zeros(grid.n_unknown))
    np.testing.assert_array_equal(config.mask, mask)
    np.testing.assert_allclose(result.image_active, 1, atol=2e-15)
    assert result.image_active.size == np.count_nonzero(~mask)
    assert grid.query_x.size == np.count_nonzero(~mask) * 9
    nodes = np.asarray(grid.node_indices)
    core = nodes[np.asarray(grid.core_indices)]
    query_corners = core[np.asarray(grid.interpolation_indices)]
    query_xy = np.column_stack((np.asarray(grid.query_x), np.asarray(grid.query_y)))
    reconstructed = np.sum(
        (query_corners * grid.spacing)
        * np.asarray(grid.interpolation_weights)[:, :, None],
        axis=1,
    )
    np.testing.assert_allclose(reconstructed, query_xy, atol=1e-16)
    all_nodes = set(map(tuple, nodes))
    for ix, iy in core:
        for offset in [-2, -1, 1, 2]:
            assert (ix + offset, iy) in all_nodes
            assert (ix, iy + offset) in all_nodes
    if selection == "isolated_edge":
        assert np.max(np.asarray(grid.x)) > np.max(np.asarray(config.xgrid))
        assert np.min(np.asarray(grid.y)) < np.min(np.asarray(config.ygrid))


@pytest.mark.parametrize("dtype", ["float64", "float32"])
def test_explicit_precision_preserves_process_defaults_and_signed_source_boundaries(
    dtype,
):
    from TinyLensGpu.ForwardSimulation.LensImage.config import SimulatorConfig
    from TinyLensGpu.ForwardSimulation.LensImage.potential_correction import (
        PotentialCorrectionRenderer,
    )
    from TinyLensGpu.PhysicalModel import PhysicalModel, Shear

    original_default = jax.config.x64_enabled
    with jax.experimental.enable_x64(dtype == "float64"):
        config = SimulatorConfig(0.1, 3, jnp.ones((1, 1)), nsub=1)
        macro = PhysicalModel(
            lens_mass=[Shear(gamma1=0.0, gamma2=0.0)], source_light=[], lens_light=[]
        )
    renderer = PotentialCorrectionRenderer(
        macro, config, 3, (-0.1, 0.1, -0.1, 0.1), dtype=dtype
    )
    signed_source = np.arange(9, dtype=np.dtype(dtype)) - 4
    grid = renderer.potential_grid
    result = renderer.render(signed_source, np.zeros(grid.n_unknown))
    displaced = renderer.render(signed_source, 0.3 * np.asarray(grid.x))
    assert jax.config.x64_enabled == original_default
    assert result.model_image.dtype == np.dtype(dtype)
    assert renderer.source_x_axis.dtype == np.dtype(dtype)
    assert grid.interpolation_weights.dtype == np.dtype(dtype)
    np.testing.assert_allclose(result.image_active, signed_source, atol=1e-6)
    np.testing.assert_array_equal(result.source_values, signed_source)
    np.testing.assert_allclose(displaced.image_active, 0, atol=1e-6)
    np.testing.assert_array_equal(displaced.source_bbox, result.source_bbox)


def test_incomplete_or_mismatched_explicit_geometry_fails_without_mask_erosion():
    from TinyLensGpu.ForwardSimulation.LensImage.config import SimulatorConfig
    from TinyLensGpu.ForwardSimulation.LensImage.potential_correction import (
        PotentialCorrectionRenderer,
    )
    from TinyLensGpu.PhysicalModel import PhysicalModel, Shear

    with jax.experimental.enable_x64():
        grid = PotentialGrid.from_queries(
            jnp.array([0.25]), jnp.array([0.75]), spacing=1.0
        )
        with pytest.raises(ValueError, match="missing.*required"):
            PotentialGrid.from_queries(
                grid.query_x,
                grid.query_y,
                spacing=1.0,
                node_indices=np.asarray(grid.node_indices)[:-1],
            )
        config = SimulatorConfig(0.1, 3, jnp.ones((1, 1)))
        macro = PhysicalModel(
            lens_mass=[Shear(gamma1=0.0, gamma2=0.0)], source_light=[], lens_light=[]
        )
        with pytest.raises(ValueError, match="actual image subpixel queries"):
            PotentialCorrectionRenderer(
                macro, config, 3, (-1.0, 1.0, -1.0, 1.0), potential_grid=grid
            )
        with pytest.raises(ValueError, match="positive"):
            PotentialCorrectionRenderer(
                macro, config, 3, (-1.0, 1.0, -1.0, 1.0), potential_spacing=0
            )
    assert not np.asarray(config.mask).any()


def test_macro_mapping_can_be_rebuilt_without_changing_fixed_geometry():
    from TinyLensGpu.ForwardSimulation.LensImage.config import SimulatorConfig
    from TinyLensGpu.ForwardSimulation.LensImage.potential_correction import (
        PotentialCorrectionRenderer,
    )
    from TinyLensGpu.PhysicalModel import PhysicalModel, Shear

    with jax.experimental.enable_x64():
        config = SimulatorConfig(0.1, 4, jnp.ones((1, 1)))
        macro = PhysicalModel(
            lens_mass=[Shear(gamma1=0.0, gamma2=0.0)], source_light=[], lens_light=[]
        )
        changed = PhysicalModel(
            lens_mass=[Shear(gamma1=0.2, gamma2=0.1)], source_light=[], lens_light=[]
        )
        renderer = PotentialCorrectionRenderer(macro, config, 3, (-1.0, 1.0, -1.0, 1.0))
        geometry = renderer.potential_grid
        updated = renderer.prepare_macro(changed)
        mapping = renderer.source_mapping(macro_mapping=updated)
    x, y = np.asarray(renderer.query_x), np.asarray(renderer.query_y)
    np.testing.assert_allclose(mapping.beta_x, 0.8 * x - 0.1 * y, atol=1e-16)
    np.testing.assert_allclose(mapping.beta_y, 1.2 * y - 0.1 * x, atol=1e-16)
    np.testing.assert_array_equal(renderer.macro_mapping.beta_x, x)
    assert renderer.potential_grid is geometry


def test_unrepresentable_potential_indices_fail_informatively():
    with pytest.raises(ValueError, match="index range"):
        PotentialGrid.from_queries(np.array([1e20]), np.array([0.0]), spacing=0.1)
