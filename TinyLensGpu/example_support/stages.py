"""Shared stage posterior persistence for example pipelines."""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from .source import resolve_source_interpolation, source_interpolation_metadata

if TYPE_CHECKING:
    from TinyLensGpu.Inference import StagePosterior


@dataclass(frozen=True)
class StageStorageConfig:
    output_dir: Path
    include_schema: bool
    source_interpolation: str | None = None


def _read_stage(path):
    with path.open("rb") as stream:
        return pickle.load(stream)


def _checkpoint_source_interpolation(extra, *, requested, path):
    records = []
    if "source_interpolation" in extra or "source_boundary" in extra:
        records.append((extra, str(path)))
    for key in ("s0", "s1"):
        package = extra.get(key)
        if isinstance(package, dict):
            records.append((package, f"{path} {key.upper()} source package"))
    for metadata, context in records or [({}, str(path))]:
        requested = resolve_source_interpolation(
            requested, metadata=metadata, context=context
        )
    return requested


def resolve_stage_source_interpolation(output_dir, tags, *, requested=None, resume=False):
    """Check all recognized checkpoints before using any saved stage results."""
    if resume:
        for tag in tags:
            path = Path(output_dir) / f"stage_{tag}.pkl"
            if path.exists():
                requested = _checkpoint_source_interpolation(
                    _read_stage(path).get("extra") or {}, requested=requested, path=path
                )
    return resolve_source_interpolation(requested)


def save_stage(
    config: StageStorageConfig, tag: str, samples, weights, param_names, log_z,
    *, extra=None, stage: StagePosterior | None = None,
) -> None:
    """Write ``stage_{tag}.pkl`` in the case's selected payload format."""
    path = config.output_dir / f"stage_{tag}.pkl"
    extra = extra or {}
    if config.source_interpolation is not None:
        extra = dict(extra)
        for key, value in source_interpolation_metadata(config.source_interpolation).items():
            extra.setdefault(key, value)
        _checkpoint_source_interpolation(extra, requested=config.source_interpolation, path=path)
    config.output_dir.mkdir(exist_ok=True)
    if config.include_schema:
        if stage is not None:
            stage_payload = stage.cache_payload()
            payload = dict(
                samples=stage_payload["samples"],
                weights=stage_payload["weights"],
                param_names=stage_payload["param_names"],
                log_z=stage_payload["log_z"],
                stage_schema={
                    key: stage_payload[key]
                    for key in ("param_names", "prior_specs")
                    if key in stage_payload
                },
                extra=extra,
            )
        else:
            payload = dict(
                samples=samples, weights=weights,
                param_names=param_names, log_z=log_z,
                stage_schema={"param_names": list(param_names)},
                extra=extra,
            )
    else:
        payload = dict(
            samples=samples, weights=weights,
            param_names=param_names, log_z=log_z,
            extra=extra,
        )
    with path.open("wb") as f:
        pickle.dump(payload, f)
    print(f"[{tag}] posterior saved to {config.output_dir}/stage_{tag}.pkl")


def load_stage(config: StageStorageConfig, tag: str) -> dict:
    """Read the stage payload, checking its provenance if a mode is selected."""
    path = config.output_dir / f"stage_{tag}.pkl"
    payload = _read_stage(path)
    if config.source_interpolation is not None:
        _checkpoint_source_interpolation(
            payload.get("extra") or {}, requested=config.source_interpolation, path=path
        )
    return payload


def stage_from_payload(payload: dict) -> StagePosterior:
    """Rehydrate a stage posterior from a cached stage payload."""
    from TinyLensGpu.Inference import StagePosterior

    schema = payload.get("stage_schema", {})
    prior_specs = payload.get("prior_specs") or schema.get("prior_specs")
    param_names = payload.get("param_names") or schema.get("param_names")
    return StagePosterior.from_schema(
        payload["samples"],
        payload["weights"],
        prior_specs=prior_specs,
        param_names=None if prior_specs is not None else param_names,
        log_z=payload.get("log_z"),
    )
