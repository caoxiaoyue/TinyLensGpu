import numpy as np
import pytest

from TinyLensGpu.Inference import ParamU
from TinyLensGpu.Inference.build_likelihood import make_likelihood
from TinyLensGpu.Inference.build_prior import make_prior_transformation
from TinyLensGpu.PhysicalModel.LensImage.composite import PhysicalModel
from TinyLensGpu.PhysicalModel.LensImage.Parametric.Light import SersicEllipse
from TinyLensGpu.ObservationModel.LensImage.multi_band_image_model import (
    BandImageData,
    BandObservationGeometry,
    MultiBandImageProbModel,
)


def _make_band(name: str) -> BandImageData:
    image = np.ones((3, 3), dtype=float)
    noise = np.ones((3, 3), dtype=float)
    psf = np.ones((3, 3), dtype=float) / 9.0
    return BandImageData(
        name=name,
        image_data=image,
        noise_map=noise,
        psf_kernel=psf,
        dpix=0.05,
        nsub=2,
        mask=None,
    )


def _make_phys_model_stub() -> PhysicalModel:
    return PhysicalModel(lens_mass=[], source_light=[], lens_light=[])


def _make_two_band_linear_model(shared_center_x: ParamU | None = None) -> tuple[MultiBandImageProbModel, ParamU]:
    if shared_center_x is None:
        shared_center_x = ParamU("shared_center_x_src", 0.0)
    shared_center_x.to_dynamic()

    phys_models = []
    for band_name in ("g", "r"):
        source = SersicEllipse(
            R_sersic=ParamU(f"{band_name}_R_sersic_src", 0.5),
            n_sersic=ParamU(f"{band_name}_n_sersic_src", 2.0),
            e1=ParamU(f"{band_name}_e1_src", 0.0),
            e2=ParamU(f"{band_name}_e2_src", 0.0),
            center_x=shared_center_x,
            center_y=ParamU(f"{band_name}_center_y_src", 0.0),
            Ie=ParamU(f"{band_name}_Ie_src", 1.0),
        )
        source.center_x.to_dynamic()
        for param in [source.R_sersic, source.n_sersic, source.e1, source.e2, source.center_y]:
            param.to_static()

        phys_models.append(PhysicalModel(lens_mass=[], source_light=[source], lens_light=[]))

    model = MultiBandImageProbModel(
        bands=[_make_band("g"), _make_band("r")],
        phys_models=phys_models,
        use_linear=True,
        solver_type="normal",
    )
    return model, shared_center_x


def _make_band_with_shape(name: str, shape: tuple[int, int], dpix: float = 0.05) -> BandImageData:
    image = np.ones(shape, dtype=float)
    noise = np.ones(shape, dtype=float)
    psf = np.ones((3, 3), dtype=float) / 9.0
    return BandImageData(
        name=name,
        image_data=image,
        noise_map=noise,
        psf_kernel=psf,
        dpix=dpix,
        nsub=2,
        mask=None,
    )


def _make_two_band_mixed_geometry_linear_model(
    shared_center_x: ParamU | None = None,
    shared_shift_x: ParamU | None = None,
) -> tuple[MultiBandImageProbModel, ParamU, ParamU]:
    if shared_center_x is None:
        shared_center_x = ParamU("shared_center_x_src", 0.0)
    if shared_shift_x is None:
        shared_shift_x = ParamU("shared_shift_x_alignment", 0.015)
    shared_center_x.to_dynamic()
    shared_shift_x.to_dynamic()

    phys_models = []
    for band_name in ("g", "r"):
        source = SersicEllipse(
            R_sersic=ParamU(f"{band_name}_R_sersic_src", 0.5),
            n_sersic=ParamU(f"{band_name}_n_sersic_src", 2.0),
            e1=ParamU(f"{band_name}_e1_src", 0.0),
            e2=ParamU(f"{band_name}_e2_src", 0.0),
            center_x=shared_center_x,
            center_y=ParamU(f"{band_name}_center_y_src", 0.0),
            Ie=ParamU(f"{band_name}_Ie_src", 1.0),
        )
        source.center_x.to_dynamic()
        for param in [source.R_sersic, source.n_sersic, source.e1, source.e2, source.center_y]:
            param.to_static()
        phys_models.append(PhysicalModel(lens_mass=[], source_light=[source], lens_light=[]))

    mask_r = np.zeros((5, 5), dtype=bool)
    mask_r[0, :] = True
    mask_r[-1, :] = True
    mask_r[:, 0] = True
    mask_r[:, -1] = True

    bands = [
        _make_band_with_shape("g", (3, 3), dpix=0.05),
        BandImageData(
            name="r",
            image_data=np.ones((5, 5), dtype=float),
            noise_map=np.ones((5, 5), dtype=float),
            psf_kernel=np.ones((3, 3), dtype=float) / 9.0,
            dpix=0.08,
            nsub=2,
            mask=mask_r,
            geometry=BandObservationGeometry(
                shift_x=shared_shift_x,
                shift_y=-0.02,
                rotation=float(np.degrees(0.12)),
                is_reference=False,
            ),
        ),
    ]

    model = MultiBandImageProbModel(
        bands=bands,
        phys_models=phys_models,
        use_linear=True,
        solver_type="normal",
    )
    return model, shared_center_x, shared_shift_x


def test_empty_band_list_raises_value_error() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        MultiBandImageProbModel(
            bands=[],
            phys_models=[],
            use_linear=True,
        )


def test_band_count_mismatch_raises_value_error() -> None:
    with pytest.raises(ValueError, match="same length"):
        MultiBandImageProbModel(
            bands=[_make_band("g")],
            phys_models=[_make_phys_model_stub(), _make_phys_model_stub()],
            use_linear=True,
        )


def test_duplicate_band_names_raise_value_error() -> None:
    with pytest.raises(ValueError, match="unique"):
        MultiBandImageProbModel(
            bands=[_make_band("r"), _make_band("r")],
            phys_models=[_make_phys_model_stub(), _make_phys_model_stub()],
            use_linear=False,
        )


def test_empty_band_name_raises_value_error() -> None:
    with pytest.raises(ValueError, match="empty"):
        MultiBandImageProbModel(
            bands=[_make_band(""), _make_band("r")],
            phys_models=[_make_phys_model_stub(), _make_phys_model_stub()],
            use_linear=False,
        )


def test_multiband_geometry_defaults_to_first_band_reference() -> None:
    bands = [
        _make_band("g"),
        BandImageData(
            name="r",
            image_data=np.ones((3, 3), dtype=float),
            noise_map=np.ones((3, 3), dtype=float),
            psf_kernel=np.ones((3, 3), dtype=float) / 9.0,
            dpix=0.05,
            nsub=2,
            mask=None,
            geometry=BandObservationGeometry(
                shift_x=0.1,
                shift_y=-0.2,
                rotation=float(np.degrees(0.3)),
                is_reference=False,
            ),
        ),
    ]
    model = MultiBandImageProbModel(
        bands=bands,
        phys_models=[_make_phys_model_stub(), _make_phys_model_stub()],
        use_linear=False,
    )

    assert model.bands[0].geometry is not None
    assert model.bands[0].geometry.is_reference is True
    assert np.isclose(model.bands[0].geometry.shift_x, 0.0)
    assert np.isclose(model.bands[0].geometry.shift_y, 0.0)
    assert np.isclose(model.bands[0].geometry.rotation, 0.0)
    assert model.bands[1].geometry is not None
    assert model.bands[1].geometry.is_reference is False


def test_multiband_rejects_multiple_reference_bands() -> None:
    bands = [
        BandImageData(
            name="g",
            image_data=np.ones((3, 3), dtype=float),
            noise_map=np.ones((3, 3), dtype=float),
            psf_kernel=np.ones((3, 3), dtype=float) / 9.0,
            dpix=0.05,
            nsub=2,
            mask=None,
            geometry=BandObservationGeometry(is_reference=True),
        ),
        BandImageData(
            name="r",
            image_data=np.ones((3, 3), dtype=float),
            noise_map=np.ones((3, 3), dtype=float),
            psf_kernel=np.ones((3, 3), dtype=float) / 9.0,
            dpix=0.05,
            nsub=2,
            mask=None,
            geometry=BandObservationGeometry(is_reference=True),
        ),
    ]

    with pytest.raises(ValueError, match="at most one"):
        MultiBandImageProbModel(
            bands=bands,
            phys_models=[_make_phys_model_stub(), _make_phys_model_stub()],
            use_linear=False,
        )


def test_multiband_alignment_anchors_reference_band() -> None:
    bands = [
        BandImageData(
            name="g",
            image_data=np.ones((3, 3), dtype=float),
            noise_map=np.ones((3, 3), dtype=float),
            psf_kernel=np.ones((3, 3), dtype=float) / 9.0,
            dpix=0.05,
            nsub=2,
            mask=None,
            geometry=BandObservationGeometry(
                shift_x=0.7,
                shift_y=-0.4,
                rotation=float(np.degrees(0.2)),
                is_reference=True,
            ),
        ),
        BandImageData(
            name="r",
            image_data=np.ones((3, 3), dtype=float),
            noise_map=np.ones((3, 3), dtype=float),
            psf_kernel=np.ones((3, 3), dtype=float) / 9.0,
            dpix=0.05,
            nsub=2,
            mask=None,
            geometry=BandObservationGeometry(
                shift_x=0.1,
                shift_y=-0.2,
                rotation=float(np.degrees(0.3)),
                is_reference=False,
            ),
        ),
    ]

    model = MultiBandImageProbModel(
        bands=bands,
        phys_models=[_make_phys_model_stub(), _make_phys_model_stub()],
        use_linear=False,
    )

    assert model.bands[0].geometry is not None
    assert model.bands[0].geometry.is_reference is True
    assert np.isclose(model.bands[0].geometry.shift_x, 0.0)
    assert np.isclose(model.bands[0].geometry.shift_y, 0.0)
    assert np.isclose(model.bands[0].geometry.rotation, 0.0)


def test_shared_alignment_params_appear_once_in_prior_specs() -> None:
    shared_shift_x = ParamU(
        "shared_shift_x_alignment",
        0.0,
        prior_type="uniform",
        prior_settings=[-1.0, 1.0],
        limits=[-5.0, 5.0],
    )
    shared_shift_x.to_dynamic()

    bands = [
        _make_band("g"),
        BandImageData(
            name="r",
            image_data=np.ones((3, 3), dtype=float),
            noise_map=np.ones((3, 3), dtype=float),
            psf_kernel=np.ones((3, 3), dtype=float) / 9.0,
            dpix=0.05,
            nsub=2,
            mask=None,
            geometry=BandObservationGeometry(shift_x=shared_shift_x, shift_y=0.0, rotation=0.0, is_reference=False),
        ),
        BandImageData(
            name="i",
            image_data=np.ones((3, 3), dtype=float),
            noise_map=np.ones((3, 3), dtype=float),
            psf_kernel=np.ones((3, 3), dtype=float) / 9.0,
            dpix=0.05,
            nsub=2,
            mask=None,
            geometry=BandObservationGeometry(shift_x=shared_shift_x, shift_y=0.0, rotation=0.0, is_reference=False),
        ),
    ]

    model = MultiBandImageProbModel(
        bands=bands,
        phys_models=[_make_phys_model_stub(), _make_phys_model_stub(), _make_phys_model_stub()],
        use_linear=False,
    )

    _, prior_specs = make_prior_transformation(model)

    assert len(prior_specs) == 1
    assert prior_specs[0].name == shared_shift_x.name


def test_multiband_dynamic_param_order_matches_caskade_order() -> None:
    model, _, _ = _make_two_band_mixed_geometry_linear_model()

    custom_names = [param.name for param in model.get_dynamic_params()]
    caskade_names = []
    seen_param_ids = set()
    for param in model.dynamic_params:
        param_id = id(param)
        if param_id in seen_param_ids:
            continue
        seen_param_ids.add(param_id)
        caskade_names.append(param.name)

    assert custom_names == caskade_names


def test_reference_band_alignment_params_are_not_exposed() -> None:
    ref_shift_x = ParamU(
        "ref_shift_x_alignment",
        0.5,
        prior_type="uniform",
        prior_settings=[-1.0, 1.0],
        limits=[-5.0, 5.0],
    )
    ref_shift_y = ParamU(
        "ref_shift_y_alignment",
        -0.3,
        prior_type="uniform",
        prior_settings=[-1.0, 1.0],
        limits=[-5.0, 5.0],
    )
    ref_rotation = ParamU(
        "ref_rotation_alignment",
        float(np.degrees(0.1)),
        prior_type="uniform",
        prior_settings=[-1.0, 1.0],
        limits=[-5.0, 5.0],
    )
    ref_shift_x.to_dynamic()
    ref_shift_y.to_dynamic()
    ref_rotation.to_dynamic()

    bands = [
        BandImageData(
            name="g",
            image_data=np.ones((3, 3), dtype=float),
            noise_map=np.ones((3, 3), dtype=float),
            psf_kernel=np.ones((3, 3), dtype=float) / 9.0,
            dpix=0.05,
            nsub=2,
            mask=None,
            geometry=BandObservationGeometry(
                shift_x=ref_shift_x,
                shift_y=ref_shift_y,
                rotation=ref_rotation,
                is_reference=True,
            ),
        ),
        _make_band("r"),
    ]

    model = MultiBandImageProbModel(
        bands=bands,
        phys_models=[_make_phys_model_stub(), _make_phys_model_stub()],
        use_linear=False,
    )

    dynamic_names = {param.name for param in model.get_dynamic_params()}

    assert ref_shift_x.name not in dynamic_names
    assert ref_shift_y.name not in dynamic_names
    assert ref_rotation.name not in dynamic_names


def test_invalid_band_geometry_raises_value_error() -> None:
    bad_band = BandImageData(
        name="g",
        image_data=np.ones((3, 3), dtype=float),
        noise_map=np.ones((2, 3), dtype=float),
        psf_kernel=np.ones((3, 3), dtype=float) / 9.0,
        dpix=0.05,
        nsub=2,
        mask=None,
    )

    with pytest.raises(ValueError, match="must match"):
        MultiBandImageProbModel(
            bands=[bad_band],
            phys_models=[_make_phys_model_stub()],
            use_linear=True,
        )


def test_band_payload_shapes_validate_before_jax() -> None:
    class ShapeOnlyArray:
        def __init__(self, shape: tuple[int, int]) -> None:
            self.shape = shape

        def __array__(self):
            raise RuntimeError("jax conversion should not run before geometry validation")

    eager_fail_band = BandImageData(
        name="g",
        image_data=ShapeOnlyArray((3, 3)),
        noise_map=np.ones((2, 2), dtype=float),
        psf_kernel=np.ones((3, 3), dtype=float) / 9.0,
        dpix=0.05,
        nsub=2,
        mask=None,
    )

    with pytest.raises(ValueError, match="must match"):
        MultiBandImageProbModel(
            bands=[eager_fail_band],
            phys_models=[_make_phys_model_stub()],
            use_linear=True,
        )


def test_multiband_allows_mixed_npix_and_dpix() -> None:
    bands = [
        _make_band_with_shape("g", (3, 3), dpix=0.05),
        _make_band_with_shape("r", (5, 5), dpix=0.08),
    ]

    model = MultiBandImageProbModel(
        bands=bands,
        phys_models=[_make_phys_model_stub(), _make_phys_model_stub()],
        use_linear=False,
    )

    assert model.band_names == ("g", "r")
    assert model.band_models[0].sim_obj.sim_config.npix == 3
    assert model.band_models[1].sim_obj.sim_config.npix == 5
    assert np.isclose(model.band_models[0].sim_obj.sim_config.dpix, 0.05)
    assert np.isclose(model.band_models[1].sim_obj.sim_config.dpix, 0.08)


def test_multiband_rejects_rectangular_band_images_for_now() -> None:
    bands = [
        _make_band_with_shape("g", (3, 3), dpix=0.05),
        _make_band_with_shape("r", (3, 4), dpix=0.08),
    ]

    with pytest.raises(ValueError, match="must be square"):
        MultiBandImageProbModel(
            bands=bands,
            phys_models=[_make_phys_model_stub(), _make_phys_model_stub()],
            use_linear=False,
        )


def test_band_names_preserve_input_order() -> None:
    bands = [
        _make_band_with_shape("i", (3, 3)),
        _make_band_with_shape("g", (3, 3)),
        _make_band_with_shape("r", (3, 3)),
    ]
    model = MultiBandImageProbModel(
        bands=bands,
        phys_models=[_make_phys_model_stub(), _make_phys_model_stub(), _make_phys_model_stub()],
        use_linear=True,
    )

    assert model.band_names == ("i", "g", "r")
    assert [band.name for band in model.bands] == ["i", "g", "r"]
    assert hasattr(model, "band_model_0")
    assert hasattr(model, "band_model_1")
    assert hasattr(model, "band_model_2")


def test_joint_loglike_equals_sum_of_single_band_models() -> None:
    model = MultiBandImageProbModel(
        bands=[_make_band("g"), _make_band("r")],
        phys_models=[_make_phys_model_stub(), _make_phys_model_stub()],
        use_linear=False,
    )

    joint_loglike = float(np.asarray(model()))
    summed_loglike = sum(float(np.asarray(band_model())) for band_model in model.band_models)

    assert np.isclose(joint_loglike, summed_loglike)


def test_masked_transformed_grid_matches_single_band_custom_grid_path() -> None:
    model, _, _ = _make_two_band_mixed_geometry_linear_model()
    theta = np.asarray(model.get_values("flat"), dtype=float)
    model.set_values(theta.tolist())

    non_identity_idx = 1
    non_identity_band_model = model.band_models[non_identity_idx]
    xgrid_sub, ygrid_sub = model._build_transformed_subgrid_1d(non_identity_idx, non_identity_band_model)

    wrapper_like = float(
        np.asarray(model._evaluate_non_identity_band_loglike(non_identity_idx, non_identity_band_model))
    )

    image_model, intensity_list = non_identity_band_model.forward_model(
        xgrid_sub=xgrid_sub,
        ygrid_sub=ygrid_sub,
    )
    direct_like = float(
        np.asarray(non_identity_band_model._evaluate_loglike_from_forward_model(image_model, intensity_list))
    )

    reference_like = float(np.asarray(model.band_models[0]()))
    joint_like = float(np.asarray(model()))

    assert xgrid_sub.shape[0] == non_identity_band_model.sim_obj.sim_config.flat_indices.shape[0]
    assert ygrid_sub.shape[0] == non_identity_band_model.sim_obj.sim_config.flat_indices.shape[0]
    assert np.isclose(wrapper_like, direct_like, rtol=1e-10, atol=1e-10)
    assert np.isclose(joint_like, reference_like + direct_like, rtol=1e-10, atol=1e-10)


def test_multiband_likelihood_returns_python_float() -> None:
    model = MultiBandImageProbModel(
        bands=[_make_band("g"), _make_band("r")],
        phys_models=[_make_phys_model_stub(), _make_phys_model_stub()],
        use_linear=False,
    )

    like = model.likelihood()
    call_like = float(np.asarray(model()))

    assert isinstance(like, float)
    assert np.isclose(like, call_like)


def test_multiband_nonfinite_joint_loglike_maps_to_finite_penalty() -> None:
    class _FakeBandModel:
        def __init__(self, value: float) -> None:
            self._value = value

        def __call__(self):
            return self._value

    model = MultiBandImageProbModel(
        bands=[_make_band("g"), _make_band("r")],
        phys_models=[_make_phys_model_stub(), _make_phys_model_stub()],
        use_linear=False,
    )

    object.__setattr__(
        model,
        "band_models",
        (_FakeBandModel(np.nan), _FakeBandModel(1.0)),
    )
    object.__setattr__(model, "_band_identity_geometry", (True, True))

    joint_loglike = float(np.asarray(model()))
    assert joint_loglike == -1e10
    assert model.likelihood() == -1e10


def test_identical_bands_contribute_twice_to_joint_loglike() -> None:
    image = np.ones((3, 3), dtype=float)
    noise = np.ones((3, 3), dtype=float)
    psf = np.ones((3, 3), dtype=float) / 9.0
    band_g = BandImageData("g", image, noise, psf, 0.05, 2, None)
    band_r = BandImageData("r", image, noise, psf, 0.05, 2, None)

    model = MultiBandImageProbModel(
        bands=[band_g, band_r],
        phys_models=[_make_phys_model_stub(), _make_phys_model_stub()],
        use_linear=False,
    )

    band_like = float(np.asarray(model.band_models[0]()))
    joint_like = float(np.asarray(model()))

    assert np.isclose(float(np.asarray(model.band_models[1]())), band_like)
    assert np.isclose(joint_like, 2.0 * band_like)


def test_multiband_import_surface() -> None:
    """Verify BandImageData and MultiBandImageProbModel are accessible from both package levels."""
    # Import from ObservationModel (public API surface)
    from TinyLensGpu.ObservationModel import BandImageData, MultiBandImageProbModel

    # Import from LensImage subpackage (deeper surface)
    from TinyLensGpu.ObservationModel.LensImage import BandImageData as B2, MultiBandImageProbModel as MB2

    # Both levels must return the same classes
    assert BandImageData is B2, "BandImageData must be identical from both import paths"
    assert MultiBandImageProbModel is MB2, "MultiBandImageProbModel must be identical from both import paths"


def test_per_band_linear_params_are_band_scoped() -> None:
    model, _ = _make_two_band_linear_model()
    theta = np.asarray(model.get_values("flat"), dtype=float)

    solved = model.get_linear_solved_params(theta.tolist())

    assert "source_light_0_g_Ie_src" in solved["g"]
    assert "source_light_0_r_Ie_src" in solved["r"]
    assert "source_light_0_g_Ie_src" not in solved["r"]
    assert "source_light_0_r_Ie_src" not in solved["g"]


def test_get_linear_solved_params_returns_nested_band_mapping() -> None:
    model, _ = _make_two_band_linear_model()
    theta = np.asarray(model.get_values("flat"), dtype=float)

    solved = model.get_linear_solved_params(theta.tolist())

    assert set(solved.keys()) == {"g", "r"}
    assert isinstance(solved["g"], dict)
    assert isinstance(solved["r"], dict)


def test_shared_param_object_reuse_survives_across_band_local_models() -> None:
    shared_center_x = ParamU("shared_center_x_src", 0.0)
    model, _ = _make_two_band_linear_model(shared_center_x=shared_center_x)
    theta = np.asarray(model.get_values("flat"), dtype=float)
    solved = model.get_linear_solved_params(theta.tolist())

    assert "center_x" in solved["g"]
    assert "center_x" in solved["r"]
    assert np.isclose(solved["g"]["center_x"], solved["r"]["center_x"])


def test_shared_params_appear_once_in_prior_specs() -> None:
    shared_center_x = ParamU(
        "shared_center_x_src",
        0.0,
        prior_type="uniform",
        prior_settings=[-1.0, 1.0],
        limits=[-5.0, 5.0],
    )
    model, shared_center_x = _make_two_band_linear_model(shared_center_x=shared_center_x)

    _, prior_specs = make_prior_transformation(model)

    assert len(prior_specs) == 1
    assert prior_specs[0].name == shared_center_x.name


def test_mixed_geometry_prior_specs_include_shared_alignment_and_physics_once() -> None:
    shared_center_x = ParamU(
        "shared_center_x_src",
        0.0,
        prior_type="uniform",
        prior_settings=[-1.0, 1.0],
        limits=[-5.0, 5.0],
    )
    shared_shift_x = ParamU(
        "shared_shift_x_alignment",
        0.015,
        prior_type="uniform",
        prior_settings=[-0.2, 0.2],
        limits=[-1.0, 1.0],
    )

    model, _, _ = _make_two_band_mixed_geometry_linear_model(
        shared_center_x=shared_center_x,
        shared_shift_x=shared_shift_x,
    )

    _, prior_specs = make_prior_transformation(model)
    prior_names = [spec.name for spec in prior_specs]

    assert prior_names.count(shared_center_x.name) == 1
    assert prior_names.count(shared_shift_x.name) == 1
    assert set(prior_names) == {shared_center_x.name, shared_shift_x.name}


def test_make_likelihood_matches_direct_wrapper_call() -> None:
    model, _ = _make_two_band_linear_model()
    theta = np.asarray(model.get_values("flat"), dtype=float)

    loglike_fn = make_likelihood(model, vectorized=False)

    like_via_helper = float(loglike_fn(theta))
    model.set_values(theta.tolist())
    like_via_direct_call = float(np.asarray(model()))

    assert np.isclose(like_via_helper, like_via_direct_call)


def test_make_likelihood_vectorized_batch_matches_manual_loop() -> None:
    model, _ = _make_two_band_linear_model()
    theta = np.asarray(model.get_values("flat"), dtype=float)
    batch = np.stack([theta, theta + 0.01, theta - 0.01], axis=0)

    loglike_fn = make_likelihood(model, vectorized=True)

    batched = np.asarray(loglike_fn(batch), dtype=float)
    manual_vals = []
    for row in batch:
        model.set_values(row.tolist())
        manual_vals.append(float(np.asarray(model())))
    manual = np.asarray(manual_vals, dtype=float)

    assert np.allclose(batched, manual)


def test_make_likelihood_vectorized_batch_matches_manual_loop_for_mixed_geometry() -> None:
    model, _, _ = _make_two_band_mixed_geometry_linear_model()
    theta = np.asarray(model.get_values("flat"), dtype=float)
    batch = np.stack([theta, theta + 0.01, theta - 0.01], axis=0)

    loglike_fn = make_likelihood(model, vectorized=True)

    batched = np.asarray(loglike_fn(batch), dtype=float)
    manual_vals = []
    for row in batch:
        model.set_values(row.tolist())
        manual_vals.append(float(np.asarray(model())))
    manual = np.asarray(manual_vals, dtype=float)

    assert np.allclose(batched, manual)


def test_make_likelihood_vectorized_accepts_prior_order_for_mixed_geometry() -> None:
    shared_center_x = ParamU(
        "shared_center_x_src",
        0.0,
        prior_type="uniform",
        prior_settings=[-1.0, 1.0],
        limits=[-5.0, 5.0],
    )
    shared_shift_x = ParamU(
        "shared_shift_x_alignment",
        0.015,
        prior_type="uniform",
        prior_settings=[-0.2, 0.2],
        limits=[-1.0, 1.0],
    )
    model, _, _ = _make_two_band_mixed_geometry_linear_model(
        shared_center_x=shared_center_x,
        shared_shift_x=shared_shift_x,
    )
    prior_transform, prior_specs = make_prior_transformation(model)
    theta = np.asarray(prior_transform(np.full(len(prior_specs), 0.5)), dtype=float)

    loglike_fn = make_likelihood(model, vectorized=True)
    loglike = float(np.asarray(loglike_fn(theta)))

    assert np.isfinite(loglike)


def test_demo_parameter_names_are_band_scoped_for_linear_terms() -> None:
    """Verify that linear amplitude parameter names carry band prefixes across 3 bands."""
    # Build 3-band model: g, r, i with SersicEllipse per band
    phys_models = []
    for band_name in ("g", "r", "i"):
        source = SersicEllipse(
            R_sersic=ParamU(f"{band_name}_R_sersic_src", 0.5),
            n_sersic=ParamU(f"{band_name}_n_sersic_src", 2.0),
            e1=ParamU(f"{band_name}_e1_src", 0.0),
            e2=ParamU(f"{band_name}_e2_src", 0.0),
            center_x=ParamU(f"{band_name}_center_x_src", 0.0),
            center_y=ParamU(f"{band_name}_center_y_src", 0.0),
            Ie=ParamU(f"{band_name}_Ie_src", 1.0),
        )
        # Only Ie is linear; all others static
        source.center_x.to_dynamic()
        for param in [source.R_sersic, source.n_sersic, source.e1, source.e2, source.center_y]:
            param.to_static()

        phys_models.append(PhysicalModel(lens_mass=[], source_light=[source], lens_light=[]))

    model = MultiBandImageProbModel(
        bands=[_make_band("g"), _make_band("r"), _make_band("i")],
        phys_models=phys_models,
        use_linear=True,
        solver_type="normal",
    )

    theta = np.asarray(model.get_values("flat"), dtype=float)
    solved = model.get_linear_solved_params(theta.tolist())

    # Each band has its own band-prefixed linear param
    assert "source_light_0_g_Ie_src" in solved["g"]
    assert "source_light_0_r_Ie_src" in solved["r"]
    assert "source_light_0_i_Ie_src" in solved["i"]

    # No cross-band leakage
    assert "source_light_0_g_Ie_src" not in solved["r"]
    assert "source_light_0_g_Ie_src" not in solved["i"]
    assert "source_light_0_r_Ie_src" not in solved["g"]
    assert "source_light_0_r_Ie_src" not in solved["i"]
    assert "source_light_0_i_Ie_src" not in solved["g"]
    assert "source_light_0_i_Ie_src" not in solved["r"]

    # Non-linear params (like center_x) may legitimately appear across bands;
    # only band-prefixed linear terms must not collide
    g_linear = {k for k in solved["g"] if k.startswith("source_light_0_g_")}
    r_linear = {k for k in solved["r"] if k.startswith("source_light_0_r_")}
    i_linear = {k for k in solved["i"] if k.startswith("source_light_0_i_")}

    assert g_linear.isdisjoint(r_linear), "g-band and r-band linear params must not collide"
    assert g_linear.isdisjoint(i_linear), "g-band and i-band linear params must not collide"
    assert r_linear.isdisjoint(i_linear), "r-band and i-band linear params must not collide"
