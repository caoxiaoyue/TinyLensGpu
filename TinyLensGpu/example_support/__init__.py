"""Shared support for maintained example pipelines (sampling, stage storage)."""

from .adaptation import (
    adapt_image_kwargs,
    artifact_fingerprint,
    pixelized_objective_provenance,
    pixelized_observation_provenance,
    recover_source_reference,
    require_reference_observation,
    source_reconstruction_provenance,
)
from .plotting import draw_image_panel
from .sampling import SamplingConfig, posterior_quantiles, run_sampling
from .source import (
    export_source_package,
    is_square_bbox,
    normalize_source_bbox,
    require_valid_reconstruction,
    resolve_source_interpolation,
    source_axes_from_bbox,
    source_bboxes_match,
    source_interpolation_metadata,
    validate_source_package,
)
from .stages import (
    StageStorageConfig,
    load_stage,
    resolve_stage_source_interpolation,
    save_stage,
    stage_from_payload,
)

__all__ = [
    "SamplingConfig",
    "StageStorageConfig",
    "adapt_image_kwargs",
    "artifact_fingerprint",
    "draw_image_panel",
    "export_source_package",
    "is_square_bbox",
    "load_stage",
    "normalize_source_bbox",
    "pixelized_objective_provenance",
    "pixelized_observation_provenance",
    "posterior_quantiles",
    "recover_source_reference",
    "require_reference_observation",
    "require_valid_reconstruction",
    "resolve_source_interpolation",
    "resolve_stage_source_interpolation",
    "run_sampling",
    "save_stage",
    "source_axes_from_bbox",
    "source_bboxes_match",
    "source_interpolation_metadata",
    "source_reconstruction_provenance",
    "stage_from_payload",
    "validate_source_package",
]
