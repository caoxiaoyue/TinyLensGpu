"""Reproduce ticket-09 small public-fit comparisons; no scientific calibration."""

import argparse
import json
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import jax
import jax.numpy as jnp
import numpy as np

from tests.test_potential_correction_iterative import _coupled_problem
from tests.test_potential_correction_iterative_operator import _changing_cells_problem


def summary(result):
    return dict(
        success=result.success,
        termination_reason=result.termination_reason,
        objective=float(result.objective),
        chi2=float(result.chi2),
        log_evidence=(
            None if result.log_evidence is None else float(result.log_evidence)
        ),
        prior_logdet=float(result.prior_logdet),
        posterior_logdet=float(result.posterior_logdet),
        accepted_iterations=result.configuration["accepted_iterations"],
        final_scaled_gradient=result.configuration["final_scaled_gradient"],
        solver_info=result.solver_info.__dict__,
        determinant_method=result.determinant_method,
    )


def errors(left, right):
    values = {}
    for field in (
        "source_values",
        "potential_values",
        "nonlinear_model_image",
        "objective",
        "deflection_x",
        "deflection_y",
        "convergence",
    ):
        values[field] = float(
            np.nanmax(
                np.abs(
                    np.asarray(getattr(left, field)) - np.asarray(getattr(right, field))
                )
            )
        )
    return values


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    record = dict(
        environment=dict(
            python=platform.python_version(),
            jax=jax.__version__,
            numpy=np.__version__,
            device=jax.devices()[0].device_kind,
        ),
        scope="small fixed-strength algorithm checks; no astrophysical recovery or strength calibration",
        cases={},
    )
    with jax.experimental.enable_x64():
        for name, problem, ls, lp, block_size in (
            ("coupled_bilinear", _coupled_problem(), 0.3, 0.02, 7),
            ("changed_cells_nsub2", _changing_cells_problem(), 1e-4, 100.0, 16),
        ):
            dense = problem.fit(ls, lp)
            operator = problem.fit(
                ls,
                lp,
                backend="operator",
                block_size=block_size,
                pcg_rtol=1e-12,
                pcg_max_iter=800,
            )
            assert dense.success and operator.success, (
                dense.termination_reason,
                operator.termination_reason,
            )
            initial = problem.linearize(problem.initial_values)
            current = problem.linearize(
                jnp.concatenate((operator.source_values, operator.potential_values))
            )
            record["cases"][name] = dict(
                n_data=problem.renderer.n_active,
                n_source=problem.n_source,
                n_potential=problem.n_potential,
                n_core=problem.renderer.potential_grid.n_core,
                nsub=problem.renderer.nsub,
                strengths=[ls, lp, problem.lambda_amplitude],
                block_size=block_size,
                block_sizes=operator.configuration["block_sizes"],
                block_partition_sha256=operator.configuration["block_partition_sha256"],
                changed_source_cells=int(
                    np.count_nonzero(
                        np.any(
                            np.asarray(initial.mapping.indices)
                            != np.asarray(current.mapping.indices),
                            axis=1,
                        )
                    )
                ),
                initial_objective=operator.history[0]["objective_before"],
                dense=summary(dense),
                operator=summary(operator),
                max_absolute_errors=errors(operator, dense),
            )
            if name == "coupled_bilinear":
                single = _coupled_problem(dtype="float32").fit(
                    ls, lp, backend="operator", block_size=block_size
                )
                assert single.success, single.termination_reason
                record["float32_coupled"] = dict(
                    result=summary(single),
                    max_absolute_errors_vs_float64_dense=errors(single, dense),
                )
    output = json.dumps(record, indent=2, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output + "\n")
    print(output)
    return record


if __name__ == "__main__":
    main()
