"""Shared support for maintained example pipelines (sampling, stage storage)."""

from .plotting import draw_image_panel
from .sampling import SamplingConfig, posterior_quantiles, run_sampling
from .source import (
    export_source_package,
    is_square_bbox,
    normalize_source_bbox,
    require_valid_reconstruction,
    source_axes_from_bbox,
    source_bboxes_match,
    validate_source_package,
)
from .stages import StageStorageConfig, load_stage, save_stage, stage_from_payload

__all__ = [
    "SamplingConfig",
    "StageStorageConfig",
    "draw_image_panel",
    "export_source_package",
    "is_square_bbox",
    "load_stage",
    "normalize_source_bbox",
    "posterior_quantiles",
    "require_valid_reconstruction",
    "run_sampling",
    "save_stage",
    "source_axes_from_bbox",
    "source_bboxes_match",
    "stage_from_payload",
    "validate_source_package",
]
