"""Shared support for maintained example pipelines (sampling, stage storage)."""

from .sampling import SamplingConfig, posterior_quantiles, run_sampling
from .stages import StageStorageConfig, load_stage, save_stage, stage_from_payload

__all__ = [
    "SamplingConfig",
    "StageStorageConfig",
    "load_stage",
    "posterior_quantiles",
    "run_sampling",
    "save_stage",
    "stage_from_payload",
]
