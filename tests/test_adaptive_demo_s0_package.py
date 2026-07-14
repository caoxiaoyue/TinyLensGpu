"""Tests for adaptive pixelized-source demo source-package validation."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import numpy as np
import pytest


def _load_model_adpt_reg():
    """Load the demo module while containing its import-time chdir."""
    repo_root = Path(__file__).resolve().parents[1]
    module_path = (
        repo_root
        / "examples"
        / "pix_src_demo_operator"
        / "pipe"
        / "no_lens_light"
        / "model_adpt_reg.py"
    )
    old_cwd = os.getcwd()
    try:
        spec = importlib.util.spec_from_file_location(
            "model_adpt_reg_source_package_validation_test", module_path
        )
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module
    finally:
        os.chdir(old_cwd)


@pytest.fixture(scope="module")
def model_adpt_reg():
    return _load_model_adpt_reg()


def _source_package(module, *, n=None, source_bbox=None, source_pixels=None):
    n = module.NSRC if n is None else int(n)
    bbox = (-0.5, 0.5, -0.5, 0.5) if source_bbox is None else source_bbox
    pixels = (
        np.ones(n * n, dtype=np.float64)
        if source_pixels is None
        else np.asarray(source_pixels, dtype=np.float64)
    )
    return {
        "source_pixels": pixels,
        "source_bbox": bbox,
        "source_x_axis": np.linspace(bbox[0], bbox[1], n),
        "source_y_axis": np.linspace(bbox[2], bbox[3], n),
        "n": n,
        "lambda_best": 1.0,
        "log_lambda_best": 0.0,
    }


@pytest.mark.unit
def test_validate_source_package_accepts_square_grid_and_bbox(model_adpt_reg):
    package = _source_package(model_adpt_reg)

    validated = model_adpt_reg._validate_source_package(package)

    assert validated is package
    assert validated["scale_map"].shape == (model_adpt_reg.NSRC * model_adpt_reg.NSRC,)
    assert np.all(np.isfinite(validated["scale_map"]))
    assert np.all(validated["scale_map"] > 0.0)


@pytest.mark.unit
def test_validate_source_package_rejects_legacy_grid_shape_metadata(model_adpt_reg):
    n = model_adpt_reg.NSRC
    bbox = (-0.5, 0.5, -0.5, 0.5)
    package = {
        "source_pixels": np.ones(n * n, dtype=np.float64),
        "source_bbox": bbox,
        "source_x_axis": np.linspace(bbox[0], bbox[1], n),
        "source_y_axis": np.linspace(bbox[2], bbox[3], n),
        "nx": n,
        "ny": n,
        "lambda_best": 1.0,
        "log_lambda_best": 0.0,
    }

    with pytest.raises(KeyError, match="legacy source-grid keys"):
        model_adpt_reg._validate_source_package(package)


@pytest.mark.unit
def test_validate_source_package_rejects_invalid_source_vector_shape(model_adpt_reg):
    package = _source_package(
        model_adpt_reg,
        source_pixels=np.ones(model_adpt_reg.NSRC * model_adpt_reg.NSRC - 1),
    )

    with pytest.raises(ValueError, match="source_pixels must have shape"):
        model_adpt_reg._validate_source_package(package)


@pytest.mark.unit
def test_validate_source_package_rejects_rectangular_source_bbox(model_adpt_reg):
    package = _source_package(
        model_adpt_reg,
        source_bbox=(-1.0, 1.0, -0.5, 0.5),
    )

    with pytest.raises(ValueError, match="source_bbox is rectangular"):
        model_adpt_reg._validate_source_package(package)
