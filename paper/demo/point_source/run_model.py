"""
End-to-end demo for point-source position modeling.

This script demonstrates:
1) Loading synthetic point-source positions from data file
2) Building PointSourceProbModel
3) Running inference with Nautilus sampler
4) Saving posterior summary and a diagnostic plot

Run from this directory:
    python run_model.py
"""

import os
import time
import gzip
import pickle
import itertools
from typing import List, Tuple

import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

from TinyLensGpu.Inference import ParamU, nautilus_posterior_summary
from TinyLensGpu.Inference.build_prior import make_prior_transformation
from TinyLensGpu.Inference.build_likelihood import make_likelihood
from TinyLensGpu.ObservationModel import PointSourceProbModel
from TinyLensGpu.PhysicalModel import PhysicalModel, SIE, Shear


def select_best_matched_positions(
    observed_positions: np.ndarray,
    predicted_candidates: np.ndarray,
    sigma_pos: np.ndarray,
) -> Tuple[np.ndarray, float]:
    """Select and order predicted image positions by global minimum chi-square."""
    observed_positions = np.asarray(observed_positions, dtype=float)
    predicted_candidates = np.asarray(predicted_candidates, dtype=float)
    sigma_pos = np.asarray(sigma_pos, dtype=float)

    n_observed = observed_positions.shape[0]
    n_pred = predicted_candidates.shape[0]
    if n_pred < n_observed:
        raise RuntimeError(
            f"Not enough predicted image positions: got {n_pred}, need {n_observed}"
        )

    best_chi2 = None
    best_ordered = None

    all_combinations = itertools.combinations(range(n_pred), n_observed)
    all_permutations = list(itertools.permutations(range(n_observed)))

    for combo in all_combinations:
        subset = predicted_candidates[list(combo)]
        for perm in all_permutations:
            ordered_subset = subset[list(perm)]
            residual = observed_positions - ordered_subset
            chi2 = float(np.sum(np.sum(residual * residual, axis=1) / (sigma_pos * sigma_pos)))
            if (best_chi2 is None) or (chi2 < best_chi2):
                best_chi2 = chi2
                best_ordered = ordered_subset

    return best_ordered, float(best_chi2)


def build_inference_model(observed_positions: np.ndarray, sigma_pos: np.ndarray) -> PointSourceProbModel:
    sie = SIE(
        theta_E=ParamU("theta_E", 1.1, prior_type="uniform", prior_settings=[0.7, 1.8], limits=[0.1, 3.0]),
        e1=0.0,
        e2=0.0,
        center_x=0.0,
        center_y=0.0,
    )
    shear = Shear(
        gamma1=ParamU("gamma1", 0.02, prior_type="uniform", prior_settings=[-0.2, 0.2], limits=[-0.5, 0.5]),
        gamma2=ParamU("gamma2", 0.00, prior_type="uniform", prior_settings=[-0.2, 0.2], limits=[-0.5, 0.5]),
    )

    sie.theta_E.to_dynamic()
    sie.e1.to_static()
    sie.e2.to_static()
    sie.center_x.to_static()
    sie.center_y.to_static()
    shear.gamma1.to_dynamic()
    shear.gamma2.to_dynamic()

    phys_model = PhysicalModel(lens_mass=[sie, shear], source_light=[], lens_light=[])

    model = PointSourceProbModel(
        phys_model=phys_model,
        observed_positions=observed_positions,
        position_sigma=sigma_pos,
        source_x=ParamU("source_x", 0.0, prior_type="uniform", prior_settings=[-0.3, 0.3], limits=[-1.0, 1.0]),
        source_y=ParamU("source_y", 0.0, prior_type="uniform", prior_settings=[-0.3, 0.3], limits=[-1.0, 1.0]),
        source_position_fixed=False,
        solver="optimization",
        solver_config={
            "initial_range": 3.0,
            "n_x": 80,
            "n_y": 80,
            "k_keep": 30,
            "num_iters": 20,
            "tolerance": 5.0e-4,
            "cluster_tol": 0.08,
            "jacobian_eps": 1.0e-6,
        },
        min_log_like=-1.0e12,
    )
    return model


def sample_with_nautilus(prior, loglike, ndim: int, param_names: List[str], n_eff: int = 600):
    from nautilus import Sampler

    sampler = Sampler(
        prior,
        loglike,
        n_dim=ndim,
        n_live=200,
        vectorized=True,
        n_batch=200,
    )
    t0 = time.time()
    sampler.run(verbose=True, n_eff=int(n_eff))
    t1 = time.time()
    print(f"Nautilus done in {t1 - t0:.2f} s")

    samples, weights, quantiles, log_z = nautilus_posterior_summary(sampler, param_names)
    return np.asarray(samples), np.asarray(weights), float(log_z), quantiles


if __name__ == "__main__":
    print("=" * 68)
    print("Point-Source Position Modeling: End-to-End Demo")
    print("=" * 68)

    os.makedirs("output", exist_ok=True)

    print("\n[Stage 1] Load synthetic point-source data")
    data_path = os.path.join("data", "point_source_positions.npz")
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            "Missing data/point_source_positions.npz. Please run 'python sim_data.py' first."
        )

    data = np.load(data_path)
    observed_positions = np.asarray(data["observed_positions"], dtype=float)
    sigma_pos = np.asarray(data["sigma_pos"], dtype=float)
    true_positions = np.asarray(data["true_positions"], dtype=float)
    source_true = np.asarray(data["source_true"], dtype=float)

    print("True source position:", source_true)
    print("True image positions:\n", true_positions)
    print("Observed image positions:\n", observed_positions)

    print("\n[Stage 2] Build inference model")
    model = build_inference_model(observed_positions=observed_positions, sigma_pos=sigma_pos)

    prior, prior_specs = make_prior_transformation(model)
    param_names = [spec.name for spec in prior_specs]
    print(f"Dynamic parameters ({len(param_names)}): {param_names}")
    loglike = make_likelihood(model, vectorized=True)

    print("\n[Stage 3] Run inference")
    n_eff = int(os.environ.get("POINT_SOURCE_DEMO_N_EFF", "600"))
    samples, weights, log_z, quantiles = sample_with_nautilus(prior, loglike, len(param_names), param_names, n_eff=n_eff)
    backend = "nautilus"

    q16_list = [float(qs[0]) for qs in quantiles.values()]
    q50_list = [float(qs[1]) for qs in quantiles.values()]
    q84_list = [float(qs[2]) for qs in quantiles.values()]
    summary = [(param_names[i], q16_list[i], q50_list[i], q84_list[i]) for i in range(len(param_names))]

    print("\nPosterior summary (q50 [q16, q84]):")
    med = {}
    for name, q16, q50, q84 in summary:
        med[name] = q50
        print(f"  {name:10s} = {q50:+.5f} [{q16:+.5f}, {q84:+.5f}]")

    theta_med = np.array([med[name] for name in param_names], dtype=float)
    model.set_values(jnp.asarray(theta_med))
    pred_candidates, _ = model.solve_image_positions()
    pred_candidates = np.asarray(pred_candidates)
    pred_img_pos, match_chi2 = select_best_matched_positions(
        observed_positions=observed_positions,
        predicted_candidates=pred_candidates,
        sigma_pos=sigma_pos,
    )

    print(f"Matched predicted points from {pred_candidates.shape[0]} candidates; match chi2 = {match_chi2:.6f}")

    print("\n[Stage 4] Save outputs")
    np.savetxt(
        "output/result_samples.csv",
        samples,
        delimiter=",",
        header=",".join(param_names),
    )

    with open("output/result_summary.csv", "w", encoding="utf-8") as f:
        f.write("parameter,q16,q50,q84\n")
        for name, q16, q50, q84 in summary:
            f.write(f"{name},{q16:.8f},{q50:.8f},{q84:.8f}\n")

    save_dict = {
        "backend": backend,
        "samples": samples,
        "weights": weights,
        "param_names": param_names,
        "summary": summary,
        "log_z": log_z,
        "observed_positions": observed_positions,
        "predicted_candidates_median": pred_candidates,
        "predicted_positions_median": pred_img_pos,
        "predicted_match_chi2_median": match_chi2,
        "true_positions": true_positions,
        "sigma_pos": sigma_pos,
        "source_true": source_true,
    }
    with gzip.open("output/results.pkl.gz", "wb") as f:
        pickle.dump(save_dict, f)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(true_positions[:, 0], true_positions[:, 1], c="tab:green", marker="o", s=70, label="True images")
    ax.scatter(observed_positions[:, 0], observed_positions[:, 1], c="tab:red", marker="x", s=80, label="Observed")
    ax.scatter(pred_img_pos[:, 0], pred_img_pos[:, 1], c="tab:blue", marker="^", s=70, label="Predicted (median)")
    ax.set_xlabel("x [arcsec]")
    ax.set_ylabel("y [arcsec]")
    ax.set_title("Point-source image positions")
    ax.legend(loc="best")
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("output/point_source_positions.png", dpi=180)
    plt.close(fig)

    print("Saved:")
    print("  output/result_samples.csv")
    print("  output/result_summary.csv")
    print("  output/results.pkl.gz")
    print("  output/point_source_positions.png")
    print("\nDone.")
