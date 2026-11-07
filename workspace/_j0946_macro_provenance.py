"""Shared source interpolation provenance for the J0946 macro drivers."""

import json

from TinyLensGpu.example_support import (
    resolve_source_interpolation,
    resolve_stage_source_interpolation,
    source_interpolation_metadata,
)


_STAGE_TAGS = ("a", "m0", "m1", "m2", "m3")
_SAMPLER_TAGS = ("a", "m1", "m2", "m3")


def resolve_run_source_interpolation(output_dir, requested, *, resume):
    """Check the manifest, sampler checkpoints, handoff, and saved stages."""
    if resume:
        manifest = output_dir / "manifest.json"
        if manifest.exists():
            requested = resolve_source_interpolation(
                requested, metadata=json.loads(manifest.read_text())["settings"],
                context=str(manifest),
            )
        for tag in _SAMPLER_TAGS:
            path = output_dir / f"sampler_{tag}.json"
            if path.exists():
                requested = resolve_source_interpolation(
                    requested, metadata=json.loads(path.read_text()), context=str(path),
                )
            elif (output_dir / f"sampler_{tag}.h5").exists():
                raise RuntimeError(f"Sampler checkpoint has no matching settings: {path}")
        handoff = output_dir / "m3_handoff.json"
        if handoff.exists():
            requested = resolve_source_interpolation(
                requested, metadata=json.loads(handoff.read_text()), context=str(handoff),
            )
    return resolve_stage_source_interpolation(
        output_dir, _STAGE_TAGS, requested=requested, resume=resume,
    )


def source_settings_match(saved, current, path):
    """Compare complete settings with the legacy bilinear default applied."""
    mode = resolve_source_interpolation(
        current["source_interpolation"], metadata=saved, context=str(path),
    )
    # Compare old settings in memory without rewriting their original metadata.
    return {**saved, **source_interpolation_metadata(mode)} == current
