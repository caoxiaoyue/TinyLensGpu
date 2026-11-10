"""Recovery checks for pixelized survey stages and their reference producers."""

from .adaptation import artifact_fingerprint, recover_source_reference


def load_pixelized_checkpoint(load_stage, output_dir, tag, *, resume):
    """Read an existing stage when the workflow requested recovery."""
    if resume and (output_dir / f"stage_{tag}.pkl").exists():
        return load_stage(tag)
    return None


def pixelized_stage_identity(payload):
    """Identify the recorded stage without including elapsed wall time."""
    extra = {key: value for key, value in payload["extra"].items() if key != "time_taken"}
    return artifact_fingerprint({**payload, "extra": extra})


def matching_pixelized_checkpoint(cached, objective, tag, *, producer=False):
    """Reuse a matching fit or let the caller verify a historical producer."""
    if cached is None:
        return None
    saved = (cached.get("extra") or {}).get("objective")
    if saved == objective or (producer and saved is None):
        return cached
    print(f"[stage-{tag.upper()}] cached objective differs; recomputing.")
    return None


def recover_pixelized_producer(likelihood, cached, source_key, medians, names):
    """Recover the reference from a verified producer without fitting pixels."""
    extra = cached["extra"]
    package = extra[source_key]
    if float(package["log_lambda_best"]) != float(medians["log_lambda_reg"]):
        raise ValueError("Saved source regularization differs from its fitted point.")
    return recover_source_reference(
        likelihood, package, medians, names,
        historical_provenance=extra.get("source_reconstruction_provenance"),
    )


def plot_cached_pixelized_stage(
    plot, output_dir, tag, likelihood, medians, names, *, producer=None,
):
    """Recreate a missing plot using the cached stage's scoring settings."""
    from TinyLensGpu.ObservationModel.LensImage.inversion_result import (
        InvalidReconstructionError,
    )

    path = output_dir / f"stage_{tag}_model.png"
    if path.exists():
        return
    try:
        if producer is not None:
            package, build = producer
            record = package["reference_provenance"]
            score = record["producer_evidence_logdet"]
            tile = record.get("producer_evidence_block_size") or likelihood.evidence_block_size
            if (score, tile) != (likelihood.evidence_logdet, likelihood.evidence_block_size):
                likelihood = build(evidence_logdet=score, evidence_block_size=tile)
        plot(f"stage-{tag.upper()}", likelihood, medians, names, str(path))
    except InvalidReconstructionError:
        raise
    except Exception as err:
        print(f"[stage-{tag.upper()}] re-plotting failed (non-fatal): {err}")
