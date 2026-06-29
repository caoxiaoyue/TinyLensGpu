"""Pixelized source light models."""

from .pixelized_source import PixelizedSourceModel


def is_pixelized_source_model(source: object) -> bool:
    """Check whether a source component carries the pixelized-source marker."""
    if hasattr(source, "is_pixelized_source"):
        return bool(getattr(source, "is_pixelized_source"))
    return isinstance(source, PixelizedSourceModel)


__all__ = ["PixelizedSourceModel", "is_pixelized_source_model"]
