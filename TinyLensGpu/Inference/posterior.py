"""Posterior processing utilities for nested-sampling results."""

import numpy as np

from TinyLensGpu.utils.misc import weighted_quantile


def nautilus_posterior_summary(
    sampler,
    param_names,
    q=(0.16, 0.50, 0.84),
    verbose=True,
):
    """Extract and summarise posterior from a finished nautilus Sampler.

    Parameters
    ----------
    sampler : nautilus.Sampler
        A nautilus ``Sampler`` that has completed ``.run()``.
    param_names : list of str
        Parameter names, one per dimension.
    q : tuple of float, optional
        Quantiles to compute (default: 0.16, 0.50, 0.84).
    verbose : bool, optional
        If True, print a formatted summary.

    Returns
    -------
    samples : np.ndarray
        Posterior samples, shape ``(n_samples, n_dim)``.
    weights : np.ndarray
        Normalised importance weights, shape ``(n_samples,)``.
    quantiles : dict
        ``{name: np.ndarray}`` mapping each parameter to its quantile values.
    log_z : float
        Log-evidence.
    """
    samples, log_w, _ = sampler.posterior()
    log_w = np.asarray(log_w)
    weights = np.exp(log_w - np.max(log_w))
    weights /= weights.sum()

    quantiles = {}
    for i, name in enumerate(param_names):
        quantiles[name] = weighted_quantile(
            np.asarray(samples[:, i]), weights, np.asarray(q),
        )

    log_z = float(np.asarray(sampler.log_z))

    if verbose:
        print("\n" + "=" * 60)
        print("Posterior Summary")
        print("=" * 60)
        for name, qs in quantiles.items():
            q16, q50, q84 = float(qs[0]), float(qs[1]), float(qs[2])
            print(f"  {name:20s} = {q50:.4f} ({q16 - q50:+.4f}, {q84 - q50:+.4f})")
        print(f"\nlog(Z) = {log_z:.2f}")

    return samples, weights, quantiles, log_z
