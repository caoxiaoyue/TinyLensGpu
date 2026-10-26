"""Shared stage posterior persistence for example pipelines."""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

from TinyLensGpu.Inference import StagePosterior


@dataclass(frozen=True)
class StageStorageConfig:
    output_dir: Path
    include_schema: bool


def save_stage(
    config: StageStorageConfig, tag: str, samples, weights, param_names, log_z,
    *, extra=None, stage: StagePosterior | None = None,
) -> None:
    """Write ``stage_{tag}.pkl`` in the case's selected payload format."""
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
                extra=extra or {},
            )
        else:
            payload = dict(
                samples=samples, weights=weights,
                param_names=param_names, log_z=log_z,
                stage_schema={"param_names": list(param_names)},
                extra=extra or {},
            )
    else:
        payload = dict(
            samples=samples, weights=weights,
            param_names=param_names, log_z=log_z,
            extra=extra or {},
        )
    with open(config.output_dir / f"stage_{tag}.pkl", "wb") as f:
        pickle.dump(payload, f)
    print(f"[{tag}] posterior saved to {config.output_dir}/stage_{tag}.pkl")


def load_stage(config: StageStorageConfig, tag: str) -> dict:
    """Read the raw ``stage_{tag}.pkl`` payload dictionary."""
    with open(config.output_dir / f"stage_{tag}.pkl", "rb") as f:
        return pickle.load(f)


def stage_from_payload(payload: dict) -> StagePosterior:
    """Rehydrate a stage posterior from a cached stage payload."""
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
