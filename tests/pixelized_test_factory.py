"""Helpers for constructing pixelized-source models in tests."""

from __future__ import annotations

from typing import Optional

from TinyLensGpu.PhysicalModel.LensImage.Pixelized import (
    PixelizedSourceConfig,
    PixelizedSourceModel,
)


def build_pixelized_source_model(
    *,
    config: Optional[PixelizedSourceConfig] = None,
    reg_scale: float = 0.05,
    reg_coefficient: float = 1.0,
) -> PixelizedSourceModel:
    """Build ``PixelizedSourceModel`` from explicit typed config objects."""
    resolved_config = config if config is not None else PixelizedSourceConfig()
    return PixelizedSourceModel(config=resolved_config, reg_scale=reg_scale, reg_coefficient=reg_coefficient)
