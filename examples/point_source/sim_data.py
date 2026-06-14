"""
Generate synthetic data for point-source position modeling demo.

Run from this directory:
    python sim_data.py
"""

import os

import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

from TinyLensGpu.ObservationModel import PointSourceProbModel
from TinyLensGpu.PhysicalModel import PhysicalModel, SIE, Shear


def build_true_model() -> PhysicalModel:
    sie = SIE(theta_E=1.20, e1=0.0, e2=0.0, center_x=0.0, center_y=0.0)
    shear = Shear(gamma1=0.04, gamma2=-0.02)

    sie.theta_E.to_static()
    sie.e1.to_static()
    sie.e2.to_static()
    sie.center_x.to_static()
    sie.center_y.to_static()
    shear.gamma1.to_static()
    shear.gamma2.to_static()

    return PhysicalModel(lens_mass=[sie, shear], source_light=[], lens_light=[])


if __name__ == "__main__":
    print("=" * 68)
    print("Point-Source Demo: Simulate Observed Image Positions")
    print("=" * 68)

    rng_seed = int(os.environ.get("POINT_SOURCE_DEMO_SEED", "2026"))

    rng = np.random.default_rng(rng_seed)
    os.makedirs("data", exist_ok=True)

    true_model = build_true_model()
    source_true = jnp.array([0.0, 0.0], dtype=jnp.float32)

    solver = PointSourceProbModel(
        phys_model=true_model,
        observed_positions=[[0.0, 0.0]],
        position_sigma=[0.01],
        source_x=float(source_true[0]),
        source_y=float(source_true[1]),
        source_position_fixed=True,
        solver="optimization",
        solver_config={
            "initial_range": 3.0,
            "n_x": 200,
            "n_y": 200,
            "k_keep": 30,
            "num_iters": 20,
            "tolerance": 5.0e-4,
            "cluster_tol": 0.08,
        },
    )

    all_images, _ = solver.solve_image_positions()
    true_positions = np.asarray(all_images)
    n_observed = true_positions.shape[0]

    if n_observed < 1:
        raise RuntimeError("Solver found no images!")

    sigma_template = np.array([0.01, 0.012, 0.015, 0.018, 0.02], dtype=float)
    if n_observed <= sigma_template.shape[0]:
        sigma_pos = sigma_template[:n_observed]
    else:
        # Extend template if we have more images than template values
        sigma_pos = np.concatenate([
            sigma_template,
            np.full((n_observed - sigma_template.shape[0],), sigma_template[-1], dtype=float)
        ])

    observed_positions = true_positions + rng.normal(0.0, sigma_pos[:, None], size=true_positions.shape)

    np.savez(
        "data/point_source_positions.npz",
        observed_positions=observed_positions,
        true_positions=true_positions,
        sigma_pos=sigma_pos,
        source_true=np.asarray(source_true),
        seed=np.array([rng_seed], dtype=np.int32),
        theta_E_true=np.array([1.20], dtype=float),
        gamma1_true=np.array([0.04], dtype=float),
        gamma2_true=np.array([-0.02], dtype=float),
    )

    np.savetxt("data/observed_positions.csv", observed_positions, delimiter=",", header="x,y")
    np.savetxt("data/true_positions.csv", true_positions, delimiter=",", header="x,y")
    np.savetxt("data/position_sigma.csv", sigma_pos, delimiter=",", header="sigma")

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(true_positions[:, 0], true_positions[:, 1], c="tab:green", marker="o", s=70, label="True")
    ax.scatter(observed_positions[:, 0], observed_positions[:, 1], c="tab:red", marker="x", s=80, label="Observed")
    ax.set_xlabel("x [arcsec]")
    ax.set_ylabel("y [arcsec]")
    ax.set_title("Simulated point-source image positions")
    ax.legend(loc="best")
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("data/point_source_data.png", dpi=180)
    plt.close(fig)

    print("Saved:")
    print("  data/point_source_positions.npz")
    print("  data/observed_positions.csv")
    print("  data/true_positions.csv")
    print("  data/position_sigma.csv")
    print("  data/point_source_data.png")
    print("\nDone.")

