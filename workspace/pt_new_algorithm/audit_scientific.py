"""Independent endpoint checks of saved full scientific experiments; no fitting."""

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np

from run_scientific import (
    ROOT,
    IterativePotentialCorrection,
    MockConfig,
    PotentialCorrectionRenderer,
    SimulatorConfig,
    generate_mock_pair,
    inspect_endpoint,
    objective_for,
    write_json,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    args = parser.parse_args()
    for run in args.runs:
        checks = {}
        config = json.loads((run / "original_configuration.json").read_text())
        canonical = json.loads(
            (
                ROOT
                / "examples/potential_correction/iterative_scientific_configuration.json"
            ).read_text()
        )
        assert (
            config == canonical
        ), "This audit is for the unchanged original 200-pixel mock"
        environment = json.loads((run / "environment.json").read_text())
        snapshot = run / "runner_snapshot.py"
        expected = environment["source_sha256"][
            "workspace/pt_new_algorithm/run_scientific.py"
        ]
        assert hashlib.sha256(snapshot.read_bytes()).hexdigest() == expected
        with jax.experimental.enable_x64(), jax.default_matmul_precision("highest"):
            mock = MockConfig.from_dict(config["mock"])
            pair = generate_mock_pair(mock)
            sim = SimulatorConfig(
                mock.dpix,
                mock.npix,
                pair.control.psf_kernel,
                config["inversion_nsub"],
                pair.control.mask,
            )
            renderer = PotentialCorrectionRenderer(
                pair.control.macro,
                sim,
                config["source_n"],
                tuple(config["source_bbox"]),
                potential_spacing=config["potential_spacing"],
                dtype=mock.dtype,
            )
            for path in sorted(run.glob("*/*/diagnostics.json")):
                directory = path.parent
                saved = json.loads(path.read_text())
                settings = json.loads((directory / "configuration.json").read_text())
                case = settings["case"]
                observation = getattr(pair, case)
                with np.load(
                    directory / "diagnostic_fields.npz", allow_pickle=False
                ) as arrays:
                    values = jnp.asarray(arrays["joint_values"])
                    source = jnp.asarray(arrays["initial_source"])
                    oldimage = np.array(arrays["nonlinear_model_image"])
                    oldkappa = np.array(arrays["convergence"])
                    np.testing.assert_allclose(
                        arrays["image_data"],
                        observation.image_data,
                        atol=1e-12,
                        rtol=1e-12,
                    )
                    np.testing.assert_array_equal(arrays["mask"], observation.mask)
                problem = IterativePotentialCorrection(
                    renderer,
                    observation.image_data,
                    observation.noise_map,
                    source,
                    lambda_amplitude=config["lambda_amplitude"],
                    source_regularization=config["source_regularization"],
                )
                strengths = config["fixed_strengths"][case]
                strengths = (strengths["lambda_source"], strengths["lambda_potential"])
                endpoint = inspect_endpoint(problem, strengths, values, panel=16)
                threshold = saved["physical_gradient_threshold"]
                assert (
                    saved["accepted_reconstruction"]
                    and saved["optimizer"]["library_success"]
                )
                assert endpoint["current_gn_scaled_gradient"] <= threshold
                assert endpoint["analytical_gn_scaled_gradient"] <= threshold
                assert endpoint["gradient_difference_in_gn_metric"] < 1e-7
                assert (
                    abs(endpoint["objective"] - saved["endpoint"]["objective"]) < 1e-7
                )
                products = problem._result_products(values, *strengths)
                np.testing.assert_allclose(
                    products["nonlinear_model_image"], oldimage, atol=1e-12, rtol=1e-12
                )
                np.testing.assert_allclose(
                    products["convergence"],
                    oldkappa,
                    atol=1e-12,
                    rtol=1e-12,
                    equal_nan=True,
                )
                comparison = saved["comparison_to_saved_lm"]
                assert abs(comparison["objective_difference"]) < 1e-4
                assert (
                    abs(
                        comparison["saved_fields_rerendered_objective"]
                        - comparison["saved_objective"]
                    )
                    < 1e-7
                )
                assert comparison["image_rms_difference_in_noise"] < 5e-4
                assert comparison["kappa_max_abs_difference"] < 1e-3
                if settings["initialization"] == "source_only":
                    assert not saved["initialization"]["source_truth_used"]
                    assert (
                        saved["initialization"]["source_equation_relative_residual"]
                        < 1e-10
                    )
                if case == "subhalo":
                    assert (
                        saved["recovery"]["peak_distance_to_injected_position_arcsec"]
                        <= config["potential_spacing"]
                    )
                else:
                    new = saved["recovery"]["convergence"]["all_core"]["recovered_rms"]
                    old = comparison["saved_lm_recovery"]["convergence"]["all_core"][
                        "recovered_rms"
                    ]
                    assert abs(new - old) < max(1e-7, old * 0.01)
                checks[f"{settings['initialization']}/{case}"] = dict(
                    passed=True,
                    endpoint=endpoint,
                    checks="canonical configuration, original gradient gates, AD/analytical gradient, saved-field rerender, same-objective LM comparison, recovery/control checks",
                )
                print(directory, endpoint, flush=True)
                jax.clear_caches()
        write_json(run / "endpoint_audit.json", checks)
        print(f"PASS: {len(checks)} audited endpoints in {run}", flush=True)


if __name__ == "__main__":
    main()
