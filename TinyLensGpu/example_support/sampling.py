"""Shared nested-sampling support for example pipelines."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from TinyLensGpu.Inference import StagePosterior


@dataclass(frozen=True)
class SamplingConfig:
    n_live: int
    n_eff: int
    vectorized: bool
    n_batch: int | None
    vectorized_chunk_size: int | None = None
    sampler_kwargs: dict = field(default_factory=dict)
    run_kwargs: dict = field(default_factory=dict)
    # Existing pipeline weights use exp(log_w); simple demos use shifted log_w.
    stabilize_weights: bool = False


def run_sampling(
    likelihood, config: SamplingConfig, *, tag: str,
    sampler_factory=None, before_sample=None, after_sample=None,
) -> StagePosterior:
    """Run nested sampling on ``likelihood`` and return a StagePosterior."""
    from TinyLensGpu.Inference import StagePosterior
    from TinyLensGpu.Inference.build_likelihood import make_likelihood
    from TinyLensGpu.Inference.build_prior import make_prior_transformation

    prior, prior_specs = make_prior_transformation(likelihood)
    param_names = [spec.name for spec in prior_specs]
    print(f"\n[{tag}] {len(param_names)} dynamic params:")
    for spec in prior_specs:
        print(f"    {spec.name:25s} {spec.describe()}")

    loglike = make_likelihood(
        likelihood, vectorized=config.vectorized,
        vectorized_chunk_size=config.vectorized_chunk_size,
    )
    if before_sample is not None:
        before_sample(prior, loglike, param_names)
    kwargs = dict(config.sampler_kwargs)
    kwargs.update(n_live=config.n_live, vectorized=config.vectorized)
    if config.n_batch is not None:
        kwargs["n_batch"] = config.n_batch
    if sampler_factory is None:
        from nautilus import Sampler
        sampler_factory = Sampler
    sampler = sampler_factory(prior, loglike, n_dim=len(param_names), **kwargs)
    sampler.run(verbose=True, n_eff=config.n_eff, **config.run_kwargs)
    samples, log_w, _ = sampler.posterior()
    samples = np.asarray(samples, dtype=np.float64)
    log_w = np.asarray(log_w, dtype=np.float64)
    weights = (
        np.exp(log_w - np.max(log_w)) if config.stabilize_weights else np.exp(log_w)
    )
    weights /= weights.sum()
    stage = StagePosterior.from_likelihood(
        likelihood, samples, weights, log_z=float(sampler.log_z),
    )
    print(f"[{tag}] log_z = {stage.log_z:.3f}")
    if after_sample is not None:
        after_sample(sampler, stage)
    return stage


def posterior_quantiles(stage: StagePosterior, q=(0.16, 0.5, 0.84)):
    """Return ``{name: quantile_array}`` like nautilus_posterior_summary."""
    from TinyLensGpu.utils.misc import weighted_quantile

    return {
        name: weighted_quantile(
            np.asarray(stage.samples)[:, i],
            np.asarray(stage.weights),
            np.asarray(q),
        )
        for i, name in enumerate(stage.param_names)
    }
