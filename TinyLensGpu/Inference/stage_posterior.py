"""Posterior-to-parameter transfer helpers for multi-stage inference."""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from .build_prior import PriorSpec, extract_prior_specs
from .param_u import ParamU
from TinyLensGpu.utils.misc import weighted_quantile


# ------------------------------------------------------------------ #
# Empirical width table
# ------------------------------------------------------------------ #
# Each entry: (width_type, value) with width_type in {"absolute", "relative"}.
#   absolute : sigma_II = value
#   relative : sigma_II = value * |posterior_median|
# Attribute keys match the semantic role, not necessarily the concrete ParamU
# name used in a stage.
_EMPIRICAL_WIDTHS: Dict[str, Dict[str, Tuple[str, float]]] = {
    "EPL": {
        "theta_E": ("relative", 0.1),
        "gamma": ("absolute", 0.2),
        "e1": ("absolute", 0.2),
        "e2": ("absolute", 0.2),
        "center_x": ("absolute", 0.1),
        "center_y": ("absolute", 0.1),
    },
    "SIE": {
        "theta_E": ("relative", 0.1),
        "e1": ("absolute", 0.2),
        "e2": ("absolute", 0.2),
        "center_x": ("absolute", 0.1),
        "center_y": ("absolute", 0.1),
    },
    "Shear": {
        "gamma1": ("absolute", 0.05),
        "gamma2": ("absolute", 0.05),
    },
    "Sersic": {
        "R_sersic": ("relative", 1.0),
        "n_sersic": ("absolute", 1.5),
        "e1": ("absolute", 0.2),
        "e2": ("absolute", 0.2),
        "center_x": ("absolute", 0.1),
        "center_y": ("absolute", 0.1),
    },
    "Gaussian": {
        "sigma": ("relative", 0.5),
        "e1": ("absolute", 0.2),
        "e2": ("absolute", 0.2),
        "center_x": ("absolute", 0.1),
        "center_y": ("absolute", 0.1),
    },
    "PixelizedSource": {
        # log_lambda_reg is stored in log space; the empirical width here is
        # applied to the log value and acts as an absolute floor.
        "log_lambda_reg": ("absolute", 1.0),
    },
}


def empirical_width(model: str, attr: str) -> Tuple[str, float]:
    """Look up the empirical width rule for ``(model, attr)``."""
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


def _coerce_prior_spec(spec) -> PriorSpec:
    if isinstance(spec, PriorSpec):
        return spec
    if isinstance(spec, dict):
        return PriorSpec(
            name=spec["name"],
            prior_type=spec["prior_type"],
            settings=tuple(spec["settings"]),
            limits=tuple(spec["limits"]) if spec.get("limits") is not None else None,
        )
    raise TypeError(f"Unsupported prior spec type: {type(spec)!r}")


class StagePosterior:
    """Posterior samples bound to a stage's dynamic-parameter schema."""

    def __init__(
        self,
        samples: np.ndarray,
        weights: np.ndarray,
        param_names: Optional[Sequence[str]] = None,
        *,
        prior_specs: Optional[Sequence[PriorSpec | dict]] = None,
        log_z: Optional[float] = None,
        factor_std: float = 5.0,
        likelihood=None,
    ) -> None:
        samples = np.asarray(samples, dtype=np.float64)
        weights = np.asarray(weights, dtype=np.float64)
        if samples.ndim != 2:
            raise ValueError("samples must be 2D with shape (n_samples, n_params)")
        if weights.ndim != 1 or weights.shape[0] != samples.shape[0]:
            raise ValueError("weights must be 1D with one entry per sample")
        weight_sum = float(weights.sum())
        if not np.isfinite(weight_sum) or weight_sum <= 0.0:
            raise ValueError("weights must have a positive finite sum")

        if prior_specs is not None:
            prior_specs = [_coerce_prior_spec(spec) for spec in prior_specs]
            names = [spec.name for spec in prior_specs]
        elif param_names is not None:
            names = list(param_names)
            prior_specs = None
        else:
            raise ValueError("Provide either prior_specs or param_names")

        if samples.shape[1] != len(names):
            raise ValueError(
                "samples column count must match the number of schema parameters "
                f"({samples.shape[1]} != {len(names)})"
            )
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ValueError(
                "Stage posterior parameter names must be unique; duplicates: "
                + ", ".join(duplicates)
            )

        self.samples = samples
        self.weights = weights / weight_sum
        self.param_names = names
        self.prior_specs = list(prior_specs) if prior_specs is not None else None
        self.log_z = None if log_z is None else float(log_z)
        self.factor_std = float(factor_std)
        self.likelihood = likelihood
        self._name_to_col = {name: i for i, name in enumerate(self.param_names)}

    @classmethod
    def from_likelihood(
        cls,
        likelihood,
        samples: np.ndarray,
        weights: np.ndarray,
        *,
        log_z: Optional[float] = None,
        factor_std: float = 5.0,
    ) -> "StagePosterior":
        """Create a stage posterior using the likelihood's prior-spec order."""
        return cls(
            samples,
            weights,
            prior_specs=extract_prior_specs(likelihood),
            log_z=log_z,
            factor_std=factor_std,
            likelihood=likelihood,
        )

    @classmethod
    def from_schema(
        cls,
        samples: np.ndarray,
        weights: np.ndarray,
        *,
        param_names: Optional[Sequence[str]] = None,
        prior_specs: Optional[Sequence[PriorSpec | dict]] = None,
        log_z: Optional[float] = None,
        factor_std: float = 5.0,
    ) -> "StagePosterior":
        """Create a stage posterior from lightweight serialized schema."""
        return cls(
            samples,
            weights,
            param_names=param_names,
            prior_specs=prior_specs,
            log_z=log_z,
            factor_std=factor_std,
        )

    def schema(self) -> dict:
        """Return lightweight schema metadata suitable for cache payloads."""
        payload = {"param_names": list(self.param_names)}
        if self.prior_specs is not None:
            payload["prior_specs"] = [
                {
                    "name": spec.name,
                    "prior_type": spec.prior_type,
                    "settings": tuple(spec.settings),
                    "limits": tuple(spec.limits) if spec.limits is not None else None,
                }
                for spec in self.prior_specs
            ]
        return payload

    def cache_payload(self) -> dict:
        """Return a lightweight posterior payload without the live likelihood."""
        return {
            "samples": self.samples,
            "weights": self.weights,
            "log_z": self.log_z,
            **self.schema(),
            "medians": self.medians(),
        }

    def _require_col(self, name: str) -> int:
        if name not in self._name_to_col:
            raise KeyError(
                f"'{name}' not found in stage posterior parameters. "
                f"Available: {self.param_names}"
            )
        return self._name_to_col[name]

    def median(self, name: str) -> float:
        """Return weighted posterior median for ``name``."""
        col = self._require_col(name)
        return float(weighted_quantile(self.samples[:, col], self.weights, 0.5))

    def std(self, name: str) -> float:
        """Return weighted posterior standard deviation for ``name``."""
        col = self._require_col(name)
        _, std = weighted_mean_std(self.samples[:, col], self.weights)
        return std

    def median_std(self, name: str) -> Tuple[float, float]:
        """Return ``(weighted_median, weighted_std)`` for ``name``."""
        return self.median(name), self.std(name)

    def medians(self) -> dict[str, float]:
        """Return weighted posterior medians keyed by parameter name."""
        return {name: self.median(name) for name in self.param_names}

    def conservative_sigma(
        self,
        name: str,
        model: str,
        attr: str,
    ) -> Tuple[float, float]:
        """Return ``(median, sigma)`` with the conservative width rule."""
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
        return med, max(sigma_I, sigma_II)

    def fixed(self, name: str, *, target: Optional[str] = None) -> ParamU:
        """Return a static ``ParamU`` fixed at the posterior median."""
        param = ParamU(target or name, self.median(name))
        param.to_static()
        return param

    def gaussian(
        self,
        name: str,
        *,
        model: str,
        attr: str,
        target: Optional[str] = None,
        limits: Optional[Sequence[float]] = None,
    ) -> ParamU:
        """Return a dynamic Gaussian-prior ``ParamU`` inherited from posterior."""
        med, sigma = self.conservative_sigma(name, model=model, attr=attr)
        param = ParamU(
            target or name,
            float(med),
            prior_type="gaussian",
            prior_settings=[float(med), float(sigma)],
            limits=list(limits) if limits is not None else None,
        )
        param.to_dynamic()
        return param
