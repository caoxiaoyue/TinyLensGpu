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
    return PhysicalModel.__new__(PhysicalModel)


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
