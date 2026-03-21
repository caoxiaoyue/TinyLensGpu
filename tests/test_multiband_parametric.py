import numpy as np
import pytest

from TinyLensGpu.Inference import ParamU
from TinyLensGpu.PhysicalModel.LensImage.composite import PhysicalModel
from TinyLensGpu.PhysicalModel.LensImage.Parametric.Light import SersicEllipse
from TinyLensGpu.ObservationModel.LensImage.multi_band_image_model import (
    BandImageData,
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


def _make_band_with_shape(name: str, shape: tuple[int, int]) -> BandImageData:
    image = np.ones(shape, dtype=float)
    noise = np.ones(shape, dtype=float)
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

    assert "g_Ie_src" in solved["g"]
    assert "r_Ie_src" in solved["r"]
    assert "g_Ie_src" not in solved["r"]
    assert "r_Ie_src" not in solved["g"]


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
