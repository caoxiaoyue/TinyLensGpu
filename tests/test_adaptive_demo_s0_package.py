"""Tests for adaptive pixelized-source demo S0 package validation."""

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
            "model_adpt_reg_s0_validation_test", module_path
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


def _s0_package(module, *, nx=None, ny=None, source_bbox=None):
    nx = module.NSRC if nx is None else int(nx)
    ny = module.NSRC if ny is None else int(ny)
    bbox = (-0.5, 0.5, -0.5, 0.5) if source_bbox is None else source_bbox
    return {
        "source_pixels": np.ones(nx * ny, dtype=np.float64),
        "source_bbox": bbox,
        "source_x_axis": np.linspace(bbox[0], bbox[1], nx),
        "source_y_axis": np.linspace(bbox[2], bbox[3], ny),
        "nx": nx,
        "ny": ny,
        "lambda_best": 1.0,
        "log_lambda_best": 0.0,
    }


@pytest.mark.unit
def test_validate_s0_package_accepts_square_grid_and_bbox(model_adpt_reg):
    package = _s0_package(model_adpt_reg)

    validated = model_adpt_reg._validate_s0_package(package)

    assert validated is package
    assert validated["scale_map"].shape == (model_adpt_reg.NSRC * model_adpt_reg.NSRC,)
    assert np.all(np.isfinite(validated["scale_map"]))
    assert np.all(validated["scale_map"] > 0.0)


@pytest.mark.unit
def test_validate_s0_package_rejects_rectangular_grid_shape(model_adpt_reg):
    package = _s0_package(
        model_adpt_reg,
        nx=model_adpt_reg.NSRC,
        ny=model_adpt_reg.NSRC - 1,
    )

    with pytest.raises(ValueError, match="rectangular"):
        model_adpt_reg._validate_s0_package(package)


@pytest.mark.unit
def test_validate_s0_package_rejects_rectangular_source_bbox(model_adpt_reg):
    package = _s0_package(
        model_adpt_reg,
        source_bbox=(-1.0, 1.0, -0.5, 0.5),
    )

    with pytest.raises(ValueError, match="source_bbox is rectangular"):
        model_adpt_reg._validate_s0_package(package)
