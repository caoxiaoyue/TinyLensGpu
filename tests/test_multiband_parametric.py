import numpy as np
import pytest

from TinyLensGpu.PhysicalModel.LensImage.composite import PhysicalModel
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
