"""Shared outputs of dense and operator pixelized-source inversions."""

from __future__ import annotations

from typing import NamedTuple

from jax import Array

from TinyLensGpu.utils.cg_solver import PCGInfo
from TinyLensGpu.utils.pnpg_solver import PNPGInfo


class InversionResult(NamedTuple):
    """One inversion's evidence and reconstruction on the returned source grid.

    ``solver_info`` contains PCG/PNPG diagnostics for the operator backend;
    it is ``None`` for the dense backend, whose current solvers expose no
    convergence status. Operator solve failures zero the reconstruction;
    dense reconstruction retains its existing numerical-failure behavior.
    ``log_evidence`` is ``None`` when only reconstruction was requested.
    """

    log_evidence: Array | None
    model_image: Array
    source_pixels: Array
    lens_amplitudes: Array | None
    source_bbox: tuple[Array, Array, Array, Array]
    solver_info: PCGInfo | PNPGInfo | None


__all__ = ["InversionResult"]
