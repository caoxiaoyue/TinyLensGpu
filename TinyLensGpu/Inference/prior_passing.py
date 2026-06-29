"""
Prior passing utility: build Gaussian priors from a previous stage posterior.

The Gaussian sigma follows the conservative rule described in the SLaM-style
pipeline: take the larger of

    sigma_I  = 5 * posterior_std
    sigma_II = empirical_width   (Absolute or Relative on the posterior median)

The empirical width table below mirrors the conventions used in the
PyAutoLens prior config YAMLs (mass/total/power_law.yaml, etc.) and is
restricted to the models used by the pix_src_pipe demo.

Usage
-----
>>> passer = GaussianPriorPasser(samples, weights, param_names)
>>> theta_E = passer.gaussian(
...     name="theta_E", model="EPL", attr="theta_E",
...     limits=[0.1, 5.0])
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from .param_u import ParamU
from TinyLensGpu.utils.misc import weighted_quantile


# ------------------------------------------------------------------ #
# Empirical width table
# ------------------------------------------------------------------ #
# Each entry: (width_type, value) with width_type in {"absolute", "relative"}.
#   absolute : sigma_II = value
#   relative : sigma_II = value * |posterior_median|
# Attribute keys match the semantic role (not the concrete ParamU name used
# in TinyLensGpu), so the same table works for SIE, EPL, etc.
_EMPIRICAL_WIDTHS: Dict[str, Dict[str, Tuple[str, float]]] = {
    "EPL": {
        "theta_E":         ("relative", 0.1),
        "gamma":           ("absolute", 0.2),
        "e1":              ("absolute", 0.2),
        "e2":              ("absolute", 0.2),
        "center_x":        ("absolute", 0.1),
        "center_y":        ("absolute", 0.1),
    },
    "SIE": {
        "theta_E":         ("relative", 0.1),
        "e1":              ("absolute", 0.2),
        "e2":              ("absolute", 0.2),
        "center_x":        ("absolute", 0.1),
        "center_y":        ("absolute", 0.1),
    },
    "Shear": {
        "gamma1":          ("absolute", 0.05),
        "gamma2":          ("absolute", 0.05),
    },
    "Sersic": {
        "R_sersic":        ("relative", 1.0),
        "n_sersic":        ("absolute", 1.5),
        "e1":              ("absolute", 0.2),
        "e2":              ("absolute", 0.2),
        "center_x":        ("absolute", 0.1),
        "center_y":        ("absolute", 0.1),
    },
    "Gaussian": {
        "sigma":           ("relative", 0.5),
        "e1":              ("absolute", 0.2),
        "e2":              ("absolute", 0.2),
        "center_x":        ("absolute", 0.1),
        "center_y":        ("absolute", 0.1),
    },
    "PixelizedSource": {
        # log_lambda_reg is stored in log space; the empirical width
        # here is applied to the log value (acts as an absolute floor).
        "log_lambda_reg":      ("absolute", 1.0),
    },
}


def empirical_width(model: str, attr: str) -> Tuple[str, float]:
    """Look up the (width_type, value) tuple for (model, attr)."""
    if model not in _EMPIRICAL_WIDTHS:
        raise KeyError(
            f"Unknown model '{model}'. Known: {sorted(_EMPIRICAL_WIDTHS)}"
        )
    table = _EMPIRICAL_WIDTHS[model]
    if attr not in table:
        raise KeyError(
            f"Unknown attr '{attr}' for model '{model}'. "
            f"Known: {sorted(table)}"
        )
    return table[attr]


def weighted_mean_std(
    samples: np.ndarray, weights: np.ndarray
) -> Tuple[float, float]:
    """Return weighted (mean, std) of a 1D sample vector."""
    w = weights / weights.sum()
    mean = float(np.sum(w * samples))
    var = float(np.sum(w * (samples - mean) ** 2))
    return mean, float(np.sqrt(max(var, 0.0)))


class GaussianPriorPasser:
    """Build Gaussian ParamU priors inherited from a weighted posterior.

    Parameters
    ----------
    samples : ndarray, shape (N, D)
        Posterior samples.
    weights : ndarray, shape (N,)
        Posterior weights (will be renormalised internally).
    param_names : sequence of str
        Column names matching ``samples``.
    factor_std : float, optional
        Multiplier applied to the posterior std when computing sigma_I.
        Default is 5.0 (the "conservative" factor in SLaM pipelines).
    """

    def __init__(
        self,
        samples: np.ndarray,
        weights: np.ndarray,
        param_names: Sequence[str],
        factor_std: float = 5.0,
    ) -> None:
        samples = np.asarray(samples, dtype=np.float64)
        weights = np.asarray(weights, dtype=np.float64)
        if samples.ndim != 2:
            raise ValueError("samples must be 2D (N, D)")
        if weights.ndim != 1 or weights.shape[0] != samples.shape[0]:
            raise ValueError("weights must be 1D with N entries")
        if samples.shape[1] != len(param_names):
            raise ValueError("param_names length must match samples columns")

        self.samples = samples
        self.weights = weights / weights.sum()
        self.param_names = list(param_names)
        self.factor_std = float(factor_std)

        self._name_to_col = {n: i for i, n in enumerate(self.param_names)}

    # -------------- basic posterior queries ------------------------- #
    def median(self, name: str) -> float:
        col = self._require_col(name)
        return float(weighted_quantile(self.samples[:, col], self.weights, 0.5))

    def std(self, name: str) -> float:
        col = self._require_col(name)
        _, s = weighted_mean_std(self.samples[:, col], self.weights)
        return s

    def median_std(self, name: str) -> Tuple[float, float]:
        return self.median(name), self.std(name)

    def _require_col(self, name: str) -> int:
        if name not in self._name_to_col:
            raise KeyError(
                f"'{name}' not found in param_names={self.param_names}"
            )
        return self._name_to_col[name]

    # -------------- sigma selection logic --------------------------- #
    def conservative_sigma(
        self,
        name: str,
        model: str,
        attr: str,
    ) -> Tuple[float, float]:
        """Return (median, sigma) with sigma = max(sigma_I, sigma_II)."""
        med, std = self.median_std(name)
        sigma_I = self.factor_std * std

        width_type, value = empirical_width(model, attr)
        width_type = width_type.lower()
        if width_type == "absolute":
            sigma_II = float(value)
        elif width_type == "relative":
            sigma_II = float(value) * abs(med)
        else:
            raise ValueError(
                f"Unknown width_type '{width_type}' (use 'absolute'/'relative')"
            )

        sigma = max(sigma_I, sigma_II)
        return med, sigma

    # -------------- ParamU factory ---------------------------------- #
    def gaussian(
        self,
        name: str,
        *,
        model: str,
        attr: str,
        limits: Optional[Sequence[float]] = None,
    ) -> ParamU:
        """Construct a Gaussian-prior ``ParamU`` from the posterior.

        Parameters
        ----------
        name : str
            Column name in the posterior (used to fetch median/std).
        model, attr : str
            Keys into the empirical width table.
        limits : sequence of 2 floats, optional
            Hard physical limits passed through to ``ParamU``.
        """
        med, sigma = self.conservative_sigma(
            name, model=model, attr=attr
        )
        return ParamU(
            name,
            float(med),
            prior_type="gaussian",
            prior_settings=[float(med), float(sigma)],
            limits=list(limits) if limits is not None else None,
        )
