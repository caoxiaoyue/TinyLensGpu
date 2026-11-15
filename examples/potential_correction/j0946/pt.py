"""Dense Float32 single-step correction of the fresh J0946 M3 macro model.

Run macro_model.py first. This driver freezes the exported M3 macro, source
mesh, source gradient and adaptive source weights. Only lambda_s,
lambda4 and lambda0 are sampled with Nautilus. Source and potential pixels
are signed linear unknowns, marginalized with the full Gaussian determinant.
No parameterized subhalo or M4/M5 result enters this analysis.
"""

from __future__ import annotations

# ruff: noqa: E402  -- configure JAX and the local checkout before importing it.

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
from functools import partial
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NVIDIA_TF32_OVERRIDE", "0")
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
from nautilus import Sampler

from TinyLensGpu.ForwardSimulation import PotentialCorrectionRenderer, SimulatorConfig
from TinyLensGpu.ObservationModel import SingleStepPotentialCorrection
from TinyLensGpu.ObservationModel.LensImage._potential_correction_common import (
    _dense_response_columns,
)
from TinyLensGpu.PhysicalModel import EPL, PhysicalModel, Shear
from TinyLensGpu.example_support import resolve_source_interpolation
from TinyLensGpu.utils.misc import weighted_quantile
from examples.potential_correction.shared.result_products import (
    RESULT_ARRAYS,
    json_safe,
    write_json,
)
from examples.potential_correction.j0946._handoff import load_handoff

HERE = Path(__file__).resolve().parent
NAMES = ["lambda_s", "lambda4", "lambda0"]
NUMERICAL_POLICY = "float32_data_with_prior_whitened_joint_QR_v2"


def potential_factor(roughness, difference, l4, l0):
    """Float32 prior root; QR preserves small λ0 before squared conditioning."""
    qp = l4 * roughness + l0 * jnp.eye(roughness.shape[0], dtype=roughness.dtype)
    diagonal = jnp.sqrt(jnp.diag(qp))

    def cholesky(_):
        return jnp.linalg.cholesky(qp / diagonal[:, None] / diagonal[None, :])

    def qr(_):
        root = jnp.concatenate(
            (
                jnp.sqrt(l4) * difference,
                jnp.sqrt(l0) * jnp.eye(roughness.shape[0], dtype=roughness.dtype),
            ),
            axis=0,
        )
        upper = jnp.linalg.qr(root / diagonal[None, :], mode="r")
        return upper.T

    factor = jax.lax.cond(l4 / l0 > 1e4, qr, cholesky, operand=None)
    return qp, diagonal, factor


def load_problem(
    macro_dir=HERE / "output" / "macro",
    potential_spacing=0.1,
    solve_rtol=1e-5,
    *,
    expected_source_interpolation=None,
):
    metadata, arrays = load_handoff(macro_dir)
    interpolation = resolve_source_interpolation(
        expected_source_interpolation, metadata=metadata, context="M3 handoff"
    )
    p = metadata["macro_parameters"]
    macro = PhysicalModel(
        lens_mass=[
            EPL(
                theta_E=p["theta_E"],
                gamma=p["gamma"],
                e1=p["e1_mass"],
                e2=p["e2_mass"],
                center_x=p["center_x_mass"],
                center_y=p["center_y_mass"],
            ),
            Shear(gamma1=p["gamma1"], gamma2=p["gamma2"]),
        ],
        source_light=[],
        lens_light=[],
    )
    config = SimulatorConfig(
        metadata["dpix"],
        arrays["mask"].shape[0],
        jnp.asarray(arrays["psf_kernel"]),
        nsub=metadata["nsub"],
        mask=arrays["mask"],
    )
    renderer = PotentialCorrectionRenderer(
        macro,
        config,
        metadata["source_n"],
        tuple(arrays["source_bbox"]),
        potential_spacing=potential_spacing,
        dtype="float32",
        source_interpolation=interpolation,
    )
    # Freeze the derivative of the saved M3 source interpolant at the macro rays.
    frozen_gradient = renderer.source_spatial_gradient(
        jnp.asarray(arrays["source_values"]),
        renderer.source_mapping(),
    )
    problem = SingleStepPotentialCorrection(
        renderer,
        arrays["lens_subtracted_image"],
        arrays["noise_map"],
        lambda bx, by: frozen_gradient,
        lambda_amplitude=1.0,
        source_regularization=metadata["source_regularization"],
        source_scale=arrays["source_scale"],
        solve_rtol=solve_rtol,
        initialization_provenance=dict(
            handoff_sha256=metadata["handoff_sha256"],
            representative=metadata["representative"],
            gradient=f"frozen_{interpolation}_derivative_of_M3_source",
            adaptive_scale=metadata.get(
                "source_scale_provenance", "fixed_M1_template_and_M2_rho_as_used_in_M3",
            ),
        ),
    )
    return problem, metadata, arrays


class DenseEvidence:
    """Cache fixed matrices for complete coupled Gaussian marginalization.

    Symmetric diagonal equilibration changes coordinates, not the prior or
    determinants. The joint/fallback path uses an augmented QR after prior
    whitening; the determinant lemma retains the full joint determinant
    without squaring the conditioning of the posterior root. All numerical
    model arrays and evidence operations are Float32, without prior jitter.
    """

    def __init__(self, problem, formulation="data"):
        self.problem = problem
        self.formulation = formulation
        print("[PT] preparing complete dense response", flush=True)
        self.design = problem.dense_mapping()
        self.design.block_until_ready()
        print(f"[PT] dense shape={self.design.shape}", flush=True)
        # Assemble the exact existing FD stencil without multiplying large
        # dense difference matrices. This is still the same complete R_s.
        self.source = jax.jit(
            lambda: _dense_response_columns(
                lambda s: problem.source_regularizer.matvec_free(
                    s, *problem.renderer.source_bbox, scale=problem.source_scale
                ),
                problem.n_source,
                problem.dtype,
            )
        )()
        self.roughness = problem.potential_regularizer.to_dense(1.0, 0.0)
        regularizer = problem.potential_regularizer
        self.difference = jnp.zeros(
            (regularizer.windows.shape[0], problem.n_potential), dtype=problem.dtype
        )
        self.difference = self.difference.at[
            jnp.arange(regularizer.windows.shape[0])[:, None], regularizer.windows
        ].set(regularizer.coefficients)
        self.source.block_until_ready()
        problem._dense_source_prior = self.source
        problem._dense_potential_roughness = self.roughness
        with jax.default_matmul_precision("highest"):
            weighted = self.design / problem.noise_active[:, None]
            self.gram = weighted.T @ weighted
            self.rhs = weighted.T @ (problem.data_active / problem.noise_active)
        diagonal = jnp.sqrt(jnp.diag(self.source))
        source_chol = jnp.linalg.cholesky(
            self.source / diagonal[:, None] / diagonal[None, :]
        )
        self.source_diagonal = diagonal
        self.source_chol = source_chol
        self.source_logdet = (
            2 * jnp.log(jnp.diag(source_chol)).sum() + 2 * jnp.log(diagonal).sum()
        )
        if not np.isfinite(float(self.source_logdet)):
            raise RuntimeError(
                "Float32 fixed source precision is not positive definite"
            )
        self.gram.block_until_ready()
        ns = problem.n_source
        self.weighted = weighted
        with jax.default_matmul_precision("highest"):
            self.source_root = jsp.linalg.solve_triangular(
                self.source_chol,
                weighted[:, :ns].T / self.source_diagonal[:, None],
                lower=True,
            )

        def solve(
            log10_strengths,
            rhs,
            weighted,
            source_factor,
            source_diagonal,
            source_root,
            roughness,
            source_logdet,
            difference,
        ):
            with jax.default_matmul_precision("highest"):
                ls, l4, l0 = jnp.power(jnp.float32(10), log10_strengths)
                _, dp, cp = potential_factor(roughness, difference, l4, l0)
                potential_root = jsp.linalg.solve_triangular(
                    cp, weighted[:, ns:].T / dp[:, None], lower=True
                )
                # Q = L L.T, V = L^-1 M.T N^-1/2. Stacking [V; I]
                # gives C = I + V.T V through QR, never through a Gram
                # product. det(H)/det(Q) = det(C) is the complete joint
                # determinant, including every source-potential cross term.
                y = problem.data_active / problem.noise_active
                augmented = jnp.concatenate(
                    (
                        source_root / jnp.sqrt(ls),
                        potential_root,
                        jnp.eye(y.size, dtype=problem.dtype),
                    ),
                    axis=0,
                )
                column_scale = jnp.sqrt(jnp.sum(augmented**2, axis=0))
                orthogonal, upper = jnp.linalg.qr(
                    augmented / column_scale[None, :], mode="reduced"
                )
                projected = jsp.linalg.solve_triangular(
                    upper.T, y / column_scale, lower=True
                )
                # V C^-1 y = Q_top R^-T D^-1 y. Use Q_top directly:
                # forming V @ dual can cancel large nearly orthogonal terms
                # before the weak-prior inverse amplifies their roundoff.
                whitened_values = orthogonal[: ns + problem.n_potential] @ projected
                source = jsp.linalg.solve_triangular(
                    source_factor.T, whitened_values[:ns], lower=False
                ) / (source_diagonal * jnp.sqrt(ls))
                potential = (
                    jsp.linalg.solve_triangular(cp.T, whitened_values[ns:], lower=False)
                    / dp
                )
                values = jnp.concatenate((source, potential))
                ldq = (
                    ns * jnp.log(ls)
                    + source_logdet
                    + 2 * (jnp.log(jnp.abs(jnp.diag(cp))).sum() + jnp.log(dp).sum())
                )
                residual = y - weighted @ values
                source_action = ls * problem.source_regularizer.matvec_free(
                    source, *problem.renderer.source_bbox, scale=problem.source_scale
                )
                potential_action = problem.potential_regularizer.matvec(
                    potential, l4, l0
                )
                differences = problem.potential_regularizer.forward(potential)
                penalty = (
                    source @ source_action
                    + l4 * (differences @ differences)
                    + l0 * (potential @ potential)
                )
                logdet_ratio = 2 * (
                    jnp.log(jnp.abs(jnp.diag(upper))).sum()
                    + jnp.log(column_scale).sum()
                )
                ldh = ldq + logdet_ratio
                chi2 = residual @ residual
                logz = -0.5 * (
                    chi2
                    + penalty
                    + logdet_ratio
                    + problem.noise_logdet
                    + problem.data_active.size * jnp.log(jnp.float32(2 * np.pi))
                )
                # Independently apply the original response and FD priors;
                # a residual of a rounded, explicitly assembled H is not a
                # certificate for this physical system or its determinant.
                normal_residual = -weighted.T @ residual + jnp.concatenate(
                    (source_action, potential_action)
                )
                relative = jnp.linalg.norm(normal_residual) / jnp.maximum(
                    jnp.linalg.norm(rhs), jnp.finfo(problem.dtype).tiny
                )
                valid = (
                    jnp.isfinite(logz)
                    & jnp.isfinite(ldq)
                    & jnp.isfinite(ldh)
                    & jnp.isfinite(relative)
                    & jnp.all(jnp.isfinite(values))
                    & (relative <= problem.solve_rtol)
                )
                return (
                    jnp.stack(
                        [
                            jnp.where(valid, logz, -jnp.inf),
                            relative,
                            chi2,
                            penalty,
                            ldq,
                            ldh,
                        ]
                    ),
                    values,
                )

        self.joint_solve = partial(
            jax.jit(solve),
            rhs=self.rhs,
            weighted=self.weighted,
            source_factor=self.source_chol,
            source_diagonal=self.source_diagonal,
            source_root=self.source_root,
            roughness=self.roughness,
            source_logdet=self.source_logdet,
            difference=self.difference,
        )
        self.solve = self.joint_solve
        if formulation == "data":
            self._prepare_data_space()
        elif formulation != "joint":
            raise ValueError("Dense formulation must be data or joint")
        self.primary_solve = self.solve

        def resolve(point):
            metrics, values = self.primary_solve(point)
            fallback = self.formulation == "data" and not np.isfinite(float(metrics[0]))
            if fallback:
                metrics, values = self.joint_solve(point)
            return (
                jnp.concatenate(
                    (metrics, jnp.asarray([fallback], dtype=problem.dtype))
                ),
                values,
            )

        self.solve = resolve

    def _prepare_data_space(self):
        """Exact Gaussian marginalization in the smaller observation space.

        C = I + F_s R_s^-1 F_s.T/lambda_s + G Q_p^-1 G.T, with noise
        whitening implicit in F_s and G. Full Dense covariance Cholesky
        preserves every source-potential coupling by the determinant lemma.
        No determinant blocks, truncated modes or iterative solves enter.
        """
        problem = self.problem
        ns = problem.n_source
        weighted = self.weighted
        with jax.default_matmul_precision("highest"):
            source_root = self.source_root
            source_covariance = source_root.T @ source_root
        potential_response = weighted[:, ns:]
        y = problem.data_active / problem.noise_active
        source_covariance.block_until_ready()

        def solve(
            point,
            source_factor,
            source_diagonal,
            source_root,
            source_covariance,
            potential_response,
            weighted,
            gram,
            rhs,
            roughness,
            source_logdet,
            difference,
        ):
            with jax.default_matmul_precision("highest"):
                ls, l4, l0 = jnp.power(jnp.float32(10), point)
                qp, dp, cp = potential_factor(roughness, difference, l4, l0)
                potential_root = jsp.linalg.solve_triangular(
                    cp,
                    potential_response.T / dp[:, None],
                    lower=True,
                )
                covariance = (
                    jnp.eye(y.size, dtype=problem.dtype)
                    + source_covariance / ls
                    + potential_root.T @ potential_root
                )
                dc = jnp.sqrt(jnp.diag(covariance))
                cc = jnp.linalg.cholesky(covariance / dc[:, None] / dc[None, :])
                dual = jsp.linalg.cho_solve((cc, True), y / dc) / dc
                source = (
                    jsp.linalg.solve_triangular(
                        source_factor.T,
                        source_root @ dual,
                        lower=False,
                    )
                    / source_diagonal
                    / ls
                )
                potential = (
                    jsp.linalg.solve_triangular(
                        cp.T,
                        potential_root @ dual,
                        lower=False,
                    )
                    / dp
                )
                values = jnp.concatenate((source, potential))
                residual = y - weighted @ values
                source_action = ls * problem.source_regularizer.matvec_free(
                    source, *problem.renderer.source_bbox, scale=problem.source_scale
                )
                potential_action = problem.potential_regularizer.matvec(
                    potential, l4, l0
                )
                prior_action = jnp.concatenate((source_action, potential_action))
                differences = problem.potential_regularizer.forward(potential)
                penalty = (
                    source @ source_action
                    + l4 * (differences @ differences)
                    + l0 * (potential @ potential)
                )
                logdet_c = 2 * (jnp.log(jnp.diag(cc)).sum() + jnp.log(dc).sum())
                ldq = (
                    ns * jnp.log(ls)
                    + source_logdet
                    + 2 * (jnp.log(jnp.abs(jnp.diag(cp))).sum() + jnp.log(dp).sum())
                )
                chi2 = residual @ residual
                logz = -0.5 * (
                    chi2
                    + penalty
                    + logdet_c
                    + problem.noise_logdet
                    + y.size * jnp.log(jnp.float32(2 * np.pi))
                )
                relative = jnp.linalg.norm(
                    gram @ values + prior_action - rhs
                ) / jnp.linalg.norm(rhs)
                valid = (
                    jnp.isfinite(logz)
                    & jnp.isfinite(relative)
                    & (relative <= problem.solve_rtol)
                )
                return (
                    jnp.stack(
                        [
                            jnp.where(valid, logz, -jnp.inf),
                            relative,
                            chi2,
                            penalty,
                            ldq,
                            ldq + logdet_c,
                        ]
                    ),
                    values,
                )

        self.solve = partial(
            jax.jit(solve),
            source_factor=self.source_chol,
            source_diagonal=self.source_diagonal,
            source_root=source_root,
            source_covariance=source_covariance,
            potential_response=potential_response,
            weighted=weighted,
            gram=self.gram,
            rhs=self.rhs,
            roughness=self.roughness,
            source_logdet=self.source_logdet,
            difference=self.difference,
        )

    def compute(self, points):
        return np.stack(
            [
                np.asarray(self.solve(jnp.asarray(point, dtype=jnp.float32))[0])
                for point in np.atleast_2d(points)
            ]
        )


def export_fit(problem, evidence, point, destination, context):
    destination.mkdir(parents=True, exist_ok=True)
    metrics, values = evidence.solve(jnp.asarray(point, dtype=jnp.float32))
    metrics = np.asarray(metrics)
    if not np.isfinite(metrics[0]):
        raise RuntimeError(f"Invalid selected Float32 fit: {metrics}")
    ls, l4, l0 = 10.0 ** np.asarray(point)
    # This change happens only outside compiled kernels, when assembling a
    # selected public result. The sampled kernel receives all three strengths.
    problem.lambda_amplitude = float(l0)
    from TinyLensGpu.ObservationModel.LensImage.potential_correction_result import (
        PotentialSolverInfo,
    )

    fallback = bool(metrics[6])
    formulation = "joint" if fallback else evidence.formulation
    info = PotentialSolverInfo(
        (
            "dense_data_covariance_cholesky"
            if formulation == "data"
            else "prior_whitened_augmented_qr"
        ),
        True,
        False,
        2 if fallback else 1,
        float(metrics[1] * jnp.linalg.norm(evidence.rhs)),
        float(metrics[1]),
        diagnostics={"fallback_float32": fallback},
    )
    result = problem.result_from_solution(
        values,
        float(ls),
        float(l4),
        log_evidence=jnp.asarray(metrics[0]),
        prior_logdet=jnp.asarray(metrics[4]),
        posterior_logdet=jnp.asarray(metrics[5]),
        solver_info=info,
        backend="dense",
        determinant_method=(
            "full_data_covariance_cholesky"
            if formulation == "data"
            else "full_gaussian_augmented_qr"
        ),
        evidence_is_exact=True,
        solver_configuration={
            "dense_formulation": formulation,
            "requested_dense_formulation": evidence.formulation,
            "fallback_float32": fallback,
            "numerical_policy": NUMERICAL_POLICY,
        },
    )
    arrays = {
        name: np.asarray(getattr(result, name))
        for name in RESULT_ARRAYS
        if getattr(result, name) is not None
    }
    arrays.update(
        potential_x=np.asarray(result.potential_grid.x),
        potential_y=np.asarray(result.potential_grid.y),
        potential_node_indices=np.asarray(result.potential_grid.node_indices),
        source_scale=np.asarray(problem.source_scale),
        mask=context["mask"],
        macro_source_values=context["source_values"],
        macro_model_image=context["macro_model_image"],
    )
    np.savez_compressed(destination / "reconstruction.npz", **arrays)
    write_json(
        destination / "result.json",
        dict(
            success=result.success,
            dtype=result.dtype,
            backend=result.backend,
            algorithm=result.algorithm,
            strengths=asdict(result.strengths),
            log_evidence=result.log_evidence,
            chi2=result.chi2,
            regularization_penalty=result.regularization_penalty,
            prior_logdet=result.prior_logdet,
            posterior_logdet=result.posterior_logdet,
            determinant_method=result.determinant_method,
            solver_info=asdict(result.solver_info),
            configuration=result.configuration,
            initialization_provenance=result.initialization_provenance,
            linear_normalized_residual_rms=np.sqrt(
                np.nanmean(np.asarray(result.linear_normalized_residual) ** 2)
            ),
            nonlinear_normalized_residual_rms=np.sqrt(
                np.nanmean(np.asarray(result.nonlinear_normalized_residual) ** 2)
            ),
            interpretation="Conditional reconstruction at the stated hyperparameters, not a detection significance or a mass measurement",
        ),
    )
    return result


def plot_diagnostics(result, context, destination):
    """Diagnostic images retain signed fields and undefined support masks."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 10, "pdf.fonttype": 42, "svg.fonttype": "none"})
    mask = context["mask"]
    # Native pixel scale is saved independently of potential grid spacing.
    # Image coordinates come from the fixed renderer, supplied in the manifest.
    manifest = json.loads((destination.parent / "manifest.json").read_text())
    half = 0.5 * mask.shape[0] * manifest["fixed_macro"]["dpix"]
    extent = (-half, half, -half, half)
    observed = context["lens_subtracted_image"]
    noise = context["noise_map"]
    macro = context["macro_model_image"]
    linear, nonlinear = np.asarray(result.linear_model_image), np.asarray(
        result.nonlinear_model_image
    )
    fig, axes = plt.subplots(2, 3, figsize=(12, 8), layout="constrained")
    vmax = np.percentile(observed[~mask], 99.5)
    residual_limit = max(
        3.0,
        np.percentile(abs(np.asarray(result.nonlinear_normalized_residual)[~mask]), 99),
    )
    for ax, image, title in zip(
        axes[0],
        [observed, macro, linear],
        ["Lens-light-subtracted data", "M3 macro prediction", "Single-step linear fit"],
    ):
        im = ax.imshow(
            np.where(mask, np.nan, image),
            origin="lower",
            extent=extent,
            cmap="viridis",
            vmin=min(0.0, np.min(observed[~mask])),
            vmax=vmax,
        )
        ax.set_title(title)
        fig.colorbar(im, ax=ax, shrink=0.8)
    for ax, image, title in zip(
        axes[1],
        [
            (observed - macro) / noise,
            (observed - linear) / noise,
            (observed - nonlinear) / noise,
        ],
        [
            "M3 residual / noise",
            "Linear residual / noise",
            "Nonlinear residual / noise",
        ],
    ):
        im = ax.imshow(
            np.where(mask, np.nan, image),
            origin="lower",
            extent=extent,
            cmap="RdBu_r",
            vmin=-residual_limit,
            vmax=residual_limit,
        )
        ax.set_title(title)
        fig.colorbar(im, ax=ax, shrink=0.8)
    selected_y, selected_x = np.nonzero(~mask)
    dpix = manifest["fixed_macro"]["dpix"]
    view_half = (
        (
            max(
                np.max(abs(selected_x - (mask.shape[1] - 1) / 2)),
                np.max(abs(selected_y - (mask.shape[0] - 1) / 2)),
            )
            + 0.5
        )
        * dpix
        * 1.06
    )
    # Show every selected image pixel, with a small margin around the arc.
    for ax in axes.flat:
        ax.set(
            xlabel="x (arcsec)",
            ylabel="y (arcsec)",
            xlim=(-view_half, view_half),
            ylim=(-view_half, view_half),
        )
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(destination / f"image_diagnostics.{suffix}", dpi=180)
    plt.close(fig)

    source = np.asarray(result.source_values).reshape(
        len(result.source_y_axis), len(result.source_x_axis)
    )
    initial = context["source_values"].reshape(source.shape)
    source_x, source_y = np.asarray(result.source_x_axis), np.asarray(
        result.source_y_axis
    )
    source_dx = float(source_x[1] - source_x[0])
    source_dy = float(source_y[1] - source_y[0])
    # Source axes locate node centers; imshow expects the outer pixel edges.
    source_extent = (
        float(source_x[0]) - source_dx / 2,
        float(source_x[-1]) + source_dx / 2,
        float(source_y[0]) - source_dy / 2,
        float(source_y[-1]) + source_dy / 2,
    )
    node = np.asarray(result.potential_grid.node_indices)
    kappa = np.full(tuple((node.max(axis=0) - node.min(axis=0) + 1)[::-1]), np.nan)
    offset = node - node.min(axis=0)
    kappa[offset[:, 1], offset[:, 0]] = np.asarray(result.convergence)
    spacing = result.potential_grid.spacing
    kappa_extent = (
        spacing * (node[:, 0].min() - 0.5),
        spacing * (node[:, 0].max() + 0.5),
        spacing * (node[:, 1].min() - 0.5),
        spacing * (node[:, 1].max() + 0.5),
    )
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), layout="constrained")
    source_max = max(float(initial.max()), float(source.max()))
    source_min = min(0.0, float(initial.min()), float(source.min()))
    for ax, image, title in zip(
        axes[:2],
        [initial, source],
        ["M3 source (gradient input)", "Jointly reconstructed source"],
    ):
        im = ax.imshow(
            image,
            origin="lower",
            extent=source_extent,
            vmin=source_min,
            vmax=source_max,
            cmap="viridis",
        )
        ax.set_title(title)
        fig.colorbar(im, ax=ax, shrink=0.8)
    limit = np.nanmax(abs(kappa))
    im = axes[2].imshow(
        kappa,
        origin="lower",
        extent=kappa_extent,
        cmap="RdBu_r",
        vmin=-limit,
        vmax=limit,
    )
    axes[2].set_title("Convergence correction")
    fig.colorbar(im, ax=axes[2], shrink=0.8, label=r"$\delta\kappa$")
    for ax in axes:
        ax.set(xlabel="x (arcsec)", ylabel="y (arcsec)")
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(destination / f"field_diagnostics.{suffix}", dpi=180)
    plt.close(fig)


def plot_posterior(points, weights, median, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 3, figsize=(8, 8), layout="constrained")
    labels = [r"$\log_{10}\lambda_s$", r"$\log_{10}\lambda_4$", r"$\log_{10}\lambda_0$"]
    for i in range(3):
        for j in range(3):
            ax = axes[i, j]
            if i < j:
                ax.set_visible(False)
                continue
            if i == j:
                ax.hist(points[:, j], bins=40, weights=weights, color="#4679a5")
                ax.axvline(median[j], color="#bb6b34")
            else:
                ax.hist2d(
                    points[:, j], points[:, i], bins=40, weights=weights, cmap="Blues"
                )
                ax.axvline(median[j], color="#bb6b34", lw=0.8)
                ax.axhline(median[i], color="#bb6b34", lw=0.8)
            ax.set_xlabel(labels[j])
            if i != j:
                ax.set_ylabel(labels[i])
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(output / f"posterior_corner.{suffix}", dpi=180)
    plt.close(fig)


def plot_saved_outputs(output, macro_dir, *, expected_source_interpolation=None):
    """Replot saved numerical results without rebuilding or solving matrices."""
    from types import SimpleNamespace

    manifest = json.loads((output / "manifest.json").read_text())
    metadata, context = load_handoff(macro_dir)
    resolve_source_interpolation(
        expected_source_interpolation, metadata=metadata, context="M3 handoff"
    )
    if manifest.get("fixed_macro") != json_safe(metadata):
        raise RuntimeError("Potential-correction fixed macro handoff differs")
    for name in ("maximum_evidence", "posterior_median"):
        with np.load(output / name / "reconstruction.npz") as archive:
            values = {key: archive[key] for key in archive.files}
        values["source_bbox"] = (
            values["source_x_axis"][0],
            values["source_x_axis"][-1],
            values["source_y_axis"][0],
            values["source_y_axis"][-1],
        )
        values["potential_grid"] = SimpleNamespace(
            node_indices=values["potential_node_indices"],
            spacing=manifest["settings"]["potential_spacing"],
        )
        plot_diagnostics(SimpleNamespace(**values), context, output / name)
    with np.load(output / "posterior.npz") as archive:
        points, weights = archive["log10_strengths"], archive["weights"]
    median = [weighted_quantile(points[:, i], weights, [0.5])[0] for i in range(3)]
    plot_posterior(points, weights, median, output)


def audit_completed_run(problem, evidence, context, output):
    """Cross-check retained samples and failed candidates entirely in Float32."""
    with np.load(output / "posterior.npz") as archive:
        points, weights, scores = (
            archive["log10_strengths"],
            archive["weights"],
            archive["log_evidence"],
        )
    weights = weights / weights.sum()
    joint = np.stack(
        [
            np.asarray(evidence.joint_solve(jnp.asarray(x, dtype=jnp.float32))[0])
            for x in points
        ]
    )
    finite = np.isfinite(joint[:, 0])
    difference = joint[finite, 0].astype(float) - scores[finite]
    w = weights[finite] / weights[finite].sum()
    logw = np.full_like(w, -np.inf)
    positive = w > 0
    logw[positive] = np.log(w[positive]) + difference[positive]
    shifted = np.exp(logw - logw.max())
    shifted /= shifted.sum()
    old_medians = [weighted_quantile(points[finite, i], w, [0.5])[0] for i in range(3)]
    new_medians = [
        weighted_quantile(points[finite, i], shifted, [0.5])[0] for i in range(3)
    ]
    center = np.sum(points * weights[:, None], axis=0)
    delta = points - center
    covariance = delta.T @ (weights[:, None] * delta)
    inverse = np.linalg.inv(covariance)
    failed = []
    noise_term = float(problem.noise_logdet) + problem.data_active.size * np.log(
        2 * np.pi
    )
    for line in (output / "evaluations.jsonl").read_text().splitlines():
        item = json.loads(line)
        if item["log_evidence"] is not None:
            continue
        x = np.asarray(item["log10_strengths"])
        primary = np.asarray(
            evidence.primary_solve(jnp.asarray(x, dtype=jnp.float32))[0]
        )
        reference = np.asarray(
            evidence.joint_solve(jnp.asarray(x, dtype=jnp.float32))[0]
        )
        raw = [
            -0.5 * (row[2] + row[3] + row[5] - row[4] + noise_term)
            for row in (primary, reference)
        ]
        failed.append(
            dict(
                log10_strengths=x,
                primary_relative_residual=primary[1],
                joint_relative_residual=reference[1],
                ungated_float32_scores=raw,
                posterior_mahalanobis_distance=np.sqrt(
                    (x - center) @ inverse @ (x - center)
                ),
            )
        )
    comparisons = []
    best = points[int(np.argmax(scores))]
    median = np.asarray(
        [weighted_quantile(points[:, i], weights, [0.5])[0] for i in range(3)]
    )
    for name, point in (("maximum_evidence", best), ("posterior_median", median)):
        ls, l4, l0 = 10.0**point
        problem.lambda_amplitude = float(l0)
        public = problem.fit(float(ls), float(l4), backend="dense")
        with np.load(output / name / "reconstruction.npz") as archive:
            fields = {
                key: archive[key]
                for key in (
                    "source_values",
                    "potential_values",
                    "convergence",
                    "derivative_valid",
                    "potential_x",
                    "potential_y",
                )
            }
        valid = fields["derivative_valid"]
        peak = np.flatnonzero(valid)[np.argmax(fields["convergence"][valid])]
        result = json.loads((output / name / "result.json").read_text())
        summary = dict(
            name=name,
            public_success=public.success,
            primary_log_evidence=result["log_evidence"],
            public_log_evidence=public.log_evidence,
            peak_kappa=float(fields["convergence"][peak]),
            peak_position_arcsec=[
                float(fields["potential_x"][peak]),
                float(fields["potential_y"][peak]),
            ],
        )
        if public.success:
            for field in ("source_values", "potential_values", "convergence"):
                a, b = fields[field], np.asarray(getattr(public, field))
                if field == "convergence":
                    a, b = a[valid], b[valid]
                summary[field + "_relative_l2_difference"] = np.linalg.norm(
                    a - b
                ) / np.linalg.norm(b)
        comparisons.append(summary)
    raw_scores = [
        v for item in failed for v in item["ungated_float32_scores"] if np.isfinite(v)
    ]
    report = dict(
        policy="Float32 in both full Dense formulations; no Float64 likelihood or solve",
        posterior_points=len(points),
        joint_valid_points=int(finite.sum()),
        joint_invalid_original_posterior_weight=float(weights[~finite].sum()),
        joint_minus_sampled_log_evidence_weighted_mean=float(w @ difference),
        joint_minus_sampled_log_evidence_weighted_rms=float(
            np.sqrt(w @ (difference**2))
        ),
        joint_minus_sampled_log_evidence_max_absolute=float(abs(difference).max()),
        importance_total_variation_on_valid_support=float(
            0.5 * np.sum(abs(shifted - w))
        ),
        original_log10_medians=old_medians,
        joint_reweighted_log10_medians=new_medians,
        unresolved_candidates=len(failed),
        minimum_failed_point_posterior_mahalanobis_distance=min(
            (row["posterior_mahalanobis_distance"] for row in failed), default=None
        ),
        largest_ungated_failed_score=max(raw_scores, default=None),
        maximum_retained_log_evidence=float(scores.max()),
        selected_map_comparisons=comparisons,
        limitations="Ungated failed scores and their distance from sampled support are diagnostics, not certificates for failed or unsampled hyperprior volume. This is not a Float64 audit or an independent sampler run.",
    )
    write_json(output / "numerical_audit.json", report)
    write_json(output / "failed_candidate_audit.json", failed)
    write_json(
        output / "joint_posterior_checks.json",
        dict(log10_strengths=points, metrics=joint),
    )
    print(json.dumps(json_safe(report)), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--macro-dir", type=Path, default=HERE / "output" / "macro")
    parser.add_argument("--output", type=Path, default=HERE / "output" / "single_step")
    parser.add_argument(
        "--source-interpolation",
        choices=("bilinear", "cubic_hermite"),
        help="Require this source interpolation in the M3 handoff (default: use its recorded mode)",
    )
    parser.add_argument("--potential-spacing", type=float, default=0.1)
    parser.add_argument("--n-live", type=int, default=300)
    parser.add_argument("--n-eff", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=947)
    parser.add_argument("--solve-rtol", type=float, default=1e-5)
    parser.add_argument("--dense-form", choices=("data", "joint"), default="data")
    parser.add_argument("--log10-lambda-s", type=float, nargs=2)
    parser.add_argument("--log10-lambda4", type=float, nargs=2)
    parser.add_argument("--log10-lambda0", type=float, nargs=2)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args(argv)
    if min(args.n_live, args.n_eff, args.batch_size) < 1:
        parser.error("sampler counts must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "manifest.json").exists() and not (
        args.resume or args.audit_only or args.plot_only
    ):
        raise RuntimeError("Existing potential-correction run: use --resume")
    if args.plot_only:
        plot_saved_outputs(
            args.output,
            args.macro_dir,
            expected_source_interpolation=args.source_interpolation,
        )
        return
    jax.config.update("jax_enable_x64", False)
    problem, metadata, context = load_problem(
        args.macro_dir,
        args.potential_spacing,
        args.solve_rtol,
        expected_source_interpolation=args.source_interpolation,
    )
    manifest_path = args.output / "manifest.json"
    saved_manifest = (
        json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    )
    # The NPZ checksum does not cover JSON settings such as the mass model
    # and source regularization. Check the complete handoff before any solve.
    if saved_manifest is not None and saved_manifest.get("fixed_macro") != json_safe(
        metadata
    ):
        raise RuntimeError("Potential-correction fixed macro handoff differs")
    if (
        saved_manifest is not None
        and saved_manifest.get("settings", {}).get("numerical_policy")
        != NUMERICAL_POLICY
    ):
        raise RuntimeError(
            "Potential-correction numerical policy differs; use a new output "
            "directory instead of mixing old checkpoint scores with the QR solver"
        )
    print(
        f"[runtime] package={__import__('TinyLensGpu').__file__}; devices={jax.devices()}",
        flush=True,
    )
    evidence = DenseEvidence(problem, formulation=args.dense_form)
    ns = problem.n_source
    response_diagonal = np.asarray(jnp.diag(evidence.gram)[ns:])
    roughness_diagonal = np.asarray(jnp.diag(evidence.roughness))
    informative = response_diagonal[response_diagonal > 0]
    smoothed = roughness_diagonal[roughness_diagonal > 0]
    if informative.size == 0 or smoothed.size == 0:
        raise RuntimeError(
            "Potential mesh has no informative response or fourth-difference windows"
        )
    scale0 = float(np.median(informative))
    scale4 = scale0 / float(np.median(smoothed))
    ls_center = np.log10(metadata["lambda_source"])
    bounds = np.asarray(
        [
            args.log10_lambda_s or [ls_center - 2, ls_center + 2],
            args.log10_lambda4 or [np.log10(scale4) - 3, np.log10(scale4) + 6],
            args.log10_lambda0 or [np.log10(scale0) - 3, np.log10(scale0) + 3],
        ]
    )
    if not np.isfinite(bounds).all() or np.any(bounds[:, 0] >= bounds[:, 1]):
        raise ValueError("Invalid log10 hyperprior bounds")
    settings = dict(
        n_live=args.n_live,
        n_eff=args.n_eff,
        n_batch=args.batch_size,
        seed=args.seed,
        solve_rtol=args.solve_rtol,
        potential_spacing=args.potential_spacing,
        log10_prior_bounds=bounds,
        handoff_sha256=metadata["handoff_sha256"],
        dtype="float32",
        backend="dense",
        dense_formulation=args.dense_form,
        f_live=0.01,
        discard_exploration=True,
        numerical_policy=NUMERICAL_POLICY,
        parameter_order=NAMES,
    )
    if saved_manifest is not None:
        if saved_manifest["settings"] != json_safe(settings):
            raise RuntimeError("Potential-correction checkpoint settings differ")
    else:
        write_json(
            manifest_path,
            dict(
                created_utc=datetime.now(timezone.utc).isoformat(),
                settings=settings,
                fixed_macro=metadata,
                source_script_sha256=hashlib.sha256(
                    Path(__file__).read_bytes()
                ).hexdigest(),
                target="Independent bounded log-uniform hyperpriors times full conditional linear Gaussian evidence",
                default_prior_rule="lambda_s: M3 +/-2 dex; lambda0: median potential data-Gram diagonal +/-3 dex; lambda4: same scale divided by median positive roughness diagonal, -3/+6 dex (expanded after the pilot hit the upper bound)",
                prior_rule_note="Empirical, data-dependent hyperprior scale choice; hyperprior-integrated evidence is not a calibrated model-comparison result",
                numerical_policy="Pure Float32. Data-space residual/factor failures use a prior-whitened augmented QR for the full joint Gaussian target; unresolved candidates return -inf. No Float64 fallback or jitter. Potential prior roots use QR when lambda4/lambda0>1e4.",
                backend_note="Complete Dense Gaussian marginalization; data-space Cholesky or augmented QR via the determinant lemma; no block determinant approximation",
            ),
        )
    if args.audit_only:
        audit_completed_run(problem, evidence, context, args.output)
        return
    rng = np.random.default_rng(946)
    center = bounds.mean(axis=1)
    probes = np.vstack(
        [
            center,
            center + np.eye(3) * 0.2,
            center - np.eye(3) * 0.2,
            rng.uniform(bounds[:, 0], bounds[:, 1], size=(12, 3)),
        ]
    )
    start = time.perf_counter()
    scores = evidence.compute(probes)
    elapsed = time.perf_counter() - start
    finite = np.isfinite(scores[:, 0])
    if not finite.any():
        write_json(args.output / "validation.json", dict(probes=probes, scores=scores))
        raise RuntimeError("No finite Float32 preflight evidence")
    best_probe = int(np.argmax(scores[:, 0]))
    checks = []
    for index in np.argsort(np.where(finite, scores[:, 0], -np.inf))[-3:]:
        ls, l4, l0 = 10.0 ** probes[index]
        problem.lambda_amplitude = float(l0)
        public = problem.fit(float(ls), float(l4), backend="dense")
        checks.append(
            dict(
                log10_strengths=probes[index],
                success=public.success,
                public_log_evidence=public.log_evidence,
                cached_log_evidence=scores[index, 0],
                difference=(
                    None
                    if not public.success
                    else float(public.log_evidence) - float(scores[index, 0])
                ),
            )
        )
        if not public.success or abs(checks[-1]["difference"]) > 0.5:
            write_json(
                args.output / "validation.json",
                dict(probes=probes, scores=scores, public_checks=checks),
            )
            raise RuntimeError(
                "Cached Float32 evidence disagrees with the public Dense fit"
            )
    warm = time.perf_counter()
    evidence.compute(probes[best_probe : best_probe + 1])
    warm = time.perf_counter() - warm
    write_json(
        args.output / "validation.json",
        dict(
            probes=probes,
            scores=scores,
            public_checks=checks,
            finite_probes=int(finite.sum()),
            total_probes=len(probes),
            preflight_seconds=elapsed,
            warm_seconds_per_candidate=warm,
            n_data=problem.renderer.n_active,
            n_source=problem.n_source,
            n_potential=problem.n_potential,
        ),
    )
    print(
        f"[PT] validation: {finite.sum()}/{len(probes)} finite; warm={warm:.3f}s/candidate",
        flush=True,
    )
    if args.validate_only:
        return
    counters = dict(calls=0, failed=0, fallback_float32=0)

    def likelihood(points):
        points = np.atleast_2d(points)
        rows = evidence.compute(points)
        counters["calls"] += len(rows)
        counters["failed"] += int((~np.isfinite(rows[:, 0])).sum())
        counters["fallback_float32"] += int(rows[:, 6].sum())
        with (args.output / "evaluations.jsonl").open("a") as stream:
            for point, row in zip(points, rows):
                stream.write(
                    json.dumps(
                        json_safe(
                            dict(
                                log10_strengths=point,
                                log_evidence=row[0],
                                relative_residual=row[1],
                                fallback_float32=bool(row[6]),
                            )
                        )
                    )
                    + "\n"
                )
        return rows[:, 0]

    checkpoint = args.output / "checkpoint.h5"
    sampler = Sampler(
        lambda u: bounds[:, 0] + np.asarray(u) * np.diff(bounds).ravel(),
        likelihood,
        n_dim=3,
        n_live=args.n_live,
        n_batch=args.batch_size,
        vectorized=True,
        seed=args.seed,
        filepath=checkpoint,
        resume=args.resume and checkpoint.exists(),
    )
    complete = False
    while not complete:
        complete = bool(
            sampler.run(
                n_eff=args.n_eff,
                f_live=0.01,
                discard_exploration=True,
                timeout=30,
                verbose=False,
            )
        )
        state = dict(
            complete=complete,
            n_like=int(sampler.n_like),
            n_eff=float(sampler.n_eff),
            hyperprior_integrated_log_evidence=float(sampler.log_z),
            current_process=counters,
        )
        write_json(args.output / "progress.json", state)
        print(json.dumps(json_safe(state)), flush=True)
    points, logw, logl = sampler.posterior()
    weights = np.exp(logw - np.max(logw))
    weights /= weights.sum()
    np.savez_compressed(
        args.output / "posterior.npz",
        log10_strengths=points,
        weights=weights,
        log_evidence=logl,
        parameter_names=NAMES,
    )
    quantiles = np.asarray(
        [weighted_quantile(points[:, i], weights, [0.16, 0.5, 0.84]) for i in range(3)]
    )
    best = points[int(np.argmax(logl))]
    median = quantiles[:, 1]
    write_json(
        args.output / "posterior_summary.json",
        dict(
            parameter_order=NAMES,
            log10_quantiles_16_50_84=quantiles,
            quantiles_16_50_84=10.0**quantiles,
            maximum_evidence_log10_strengths=best,
            maximum_conditional_log_evidence=float(np.max(logl)),
            hyperprior_integrated_log_evidence=float(sampler.log_z),
            n_eff=float(sampler.n_eff),
            n_like=int(sampler.n_like),
            n_samples=len(points),
            posterior_weight_outer_2pct=[
                float(
                    weights[
                        (points[:, i] < bounds[i, 0] + 0.02 * np.diff(bounds[i])[0])
                        | (points[:, i] > bounds[i, 1] - 0.02 * np.diff(bounds[i])[0])
                    ].sum()
                )
                for i in range(3)
            ],
        ),
    )
    best_result = export_fit(
        problem, evidence, best, args.output / "maximum_evidence", context
    )
    plot_diagnostics(best_result, context, args.output / "maximum_evidence")
    median_result = export_fit(
        problem, evidence, median, args.output / "posterior_median", context
    )
    plot_diagnostics(median_result, context, args.output / "posterior_median")
    plot_posterior(points, weights, median, args.output)
    write_json(
        args.output / "completion.json",
        dict(
            complete=True,
            n_like=int(sampler.n_like),
            n_eff=float(sampler.n_eff),
            numerical_policy="Float32 only",
            current_process=counters,
        ),
    )
    print("[PT] complete", flush=True)


if __name__ == "__main__":
    main()
