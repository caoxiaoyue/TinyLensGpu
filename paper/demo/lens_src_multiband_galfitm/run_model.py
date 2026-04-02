# pyright: reportMissingImports=false

from __future__ import annotations

import gzip
import os
import pickle
import time
from pathlib import Path

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from nautilus import Sampler

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import TinyLensGpu
import caskade as ck
from TinyLensGpu.Inference import ParamU
from TinyLensGpu.Inference.build_likelihood import make_likelihood
from TinyLensGpu.Inference.build_prior import make_prior_transformation
from TinyLensGpu.ObservationModel import BandImageData, MultiBandImageProbModel
from TinyLensGpu.ObservationModel.LensImage.multi_band_image_model import BandObservationGeometry
from TinyLensGpu.PhysicalModel.LensImage.Parametric.Light import SersicEllipse
from TinyLensGpu.PhysicalModel.LensImage.Parametric.Mass import SIE
from TinyLensGpu.PhysicalModel.LensImage.composite import PhysicalModel
from TinyLensGpu.utils import load_lens_data
from TinyLensGpu.utils.chebyshev import (
    chebyshev_node,
    chebyshev_polynomial,
    compute_wavelength_range,
)


BANDS = ("g", "r", "i")

# Toggle to fit alignment parameters for non-reference bands
FIT_ALIGNMENT_PARAMS = True

# Known apparent image misalignment injected in sim_data.py (arcsec for shifts, degrees for rotation)
BAND_ALIGNMENTS = {
    "g": {"shift_x": 0.0, "shift_y": 0.0, "rotation": 0.0},  # Reference band
    "r": {"shift_x": 0.02, "shift_y": -0.015, "rotation": 0.573},
    "i": {"shift_x": 0.0, "shift_y": 0.0, "rotation": 0.0},
}
ROTATION_PRIOR_SIGMA_DEG = float(np.degrees(0.02))
ROTATION_LIMIT_DEG = float(np.degrees(0.1))


def evaluate_chebyshev_at_z(z: float, coeffs: list[float]) -> float:
    """Evaluate Chebyshev series at normalized wavelength z."""
    result = 0.0
    for order, coeff in enumerate(coeffs):
        result += coeff * float(chebyshev_polynomial(z, order))
    return result


def build_chebyshev_coeffs_param(name: str, initial_guess: list[float], is_radius: bool = True) -> list[ParamU]:
    """
    Build ParamU objects for Chebyshev coefficients.
    
    For 2nd-order Chebyshev, we have 3 coefficients: c0, c1, c2
    representing: p(z) = c0*T0(z) + c1*T1(z) + c2*T2(z)
    """
    coeffs = []
    for i, val in enumerate(initial_guess):
        if is_radius:
            # R_sersic coefficients: c0 > 0, c1/c2 can vary
            if i == 0:
                limits = [0.01, 5.0]
            else:
                limits = [-1.0, 1.0]
        else:
            # n_sersic coefficients: c0 in valid Sersic range, c1/c2 can vary
            if i == 0:
                limits = [0.3, 6.0]
            else:
                limits = [-1.0, 1.0]
        coeff = ParamU(f"{name}_c{i}", val, prior_type="gaussian", prior_settings=[val, 0.5], limits=limits)
        coeff.to_dynamic()
        coeffs.append(coeff)
    return coeffs


def build_shared_chebyshev_params(lambda_min: float, lambda_max: float) -> dict:
    """
    Build shared parameters using Chebyshev polynomial evolution.
    
    Following GALFITM method:
    - Mass parameters (theta_E, e1_mass, e2_mass) are shared across bands
    - Position and ellipticity (center_x, center_y, e1, e2) are constant across bands
    - R_sersic and n_sersic evolve with wavelength via Chebyshev polynomials
      (represented by 3 coefficients each: c0, c1, c2 for 2nd-order)
    
    Returns
    -------
    params : dict
        Dictionary containing all shared parameters and Chebyshev coefficients
    """
    params = {}
    
    # Mass parameters (shared across all bands)
    params["theta_E"] = ParamU("theta_E", 1.5, prior_type="uniform", prior_settings=[0.001, 3.001], limits=[0.0, 10.0])
    params["theta_E"].to_dynamic()
    
    # Lens position and ellipticity (shared between mass and light, constant across bands)
    params["e1_lens"] = ParamU("e1_lens", 0.0, prior_type="gaussian", prior_settings=[0.0, 0.3], limits=[-1.0, 1.0])
    params["e1_lens"].to_dynamic()
    params["e2_lens"] = ParamU("e2_lens", 0.0, prior_type="gaussian", prior_settings=[0.0, 0.3], limits=[-1.0, 1.0])
    params["e2_lens"].to_dynamic()
    params["center_x_lens"] = ParamU("center_x_lens", 0.0, prior_type="gaussian", prior_settings=[0.0, 0.1], limits=[-1.0, 1.0])
    params["center_x_lens"].to_dynamic()
    params["center_y_lens"] = ParamU("center_y_lens", 0.0, prior_type="gaussian", prior_settings=[0.0, 0.1], limits=[-1.0, 1.0])
    params["center_y_lens"].to_dynamic()
    
    # Source galaxy Chebyshev coefficients for R_sersic and n_sersic
    # Initial guesses based on simulation values
    # R_sersic: c0=0.35, c1=-0.05, c2=0.02
    params["R_sersic_src_coeffs"] = build_chebyshev_coeffs_param("R_sersic_src", [0.35, -0.05, 0.02], is_radius=True)
    # n_sersic: c0=0.9, c1=0.15, c2=-0.03
    params["n_sersic_src_coeffs"] = build_chebyshev_coeffs_param("n_sersic_src", [0.9, 0.15, -0.03], is_radius=False)
    
    # Source position and ellipticity (constant across bands)
    params["e1_src"] = ParamU("e1_src", 0.0, prior_type="gaussian", prior_settings=[0.0, 0.3], limits=[-1.0, 1.0])
    params["e1_src"].to_dynamic()
    params["e2_src"] = ParamU("e2_src", 0.0, prior_type="gaussian", prior_settings=[0.0, 0.3], limits=[-1.0, 1.0])
    params["e2_src"].to_dynamic()
    params["center_x_src"] = ParamU("center_x_src", 0.0, prior_type="gaussian", prior_settings=[0.0, 0.5], limits=[-3.0, 3.0])
    params["center_x_src"].to_dynamic()
    params["center_y_src"] = ParamU("center_y_src", 0.3, prior_type="gaussian", prior_settings=[0.3, 0.5], limits=[-3.0, 3.0])
    params["center_y_src"].to_dynamic()
    
    # Lens galaxy Chebyshev coefficients for R_sersic and n_sersic
    # R_sersic: c0=1.05, c1=0.02, c2=-0.01
    params["R_sersic_lens_coeffs"] = build_chebyshev_coeffs_param("R_sersic_lens", [1.05, 0.02, -0.01], is_radius=True)
    # n_sersic: c0=4.1, c1=-0.1, c2=0.05
    params["n_sersic_lens_coeffs"] = build_chebyshev_coeffs_param("n_sersic_lens", [4.1, -0.1, 0.05], is_radius=False)
    
    return params


def build_band_physical_model(
    band: str,
    shared: dict[str, ParamU],
    z: float,
) -> PhysicalModel:
    """
    Build physical model for a specific band using Chebyshev-linked parameters.
    
    Uses caskade's parameter linking to compute R_sersic and n_sersic at the
    band's wavelength from the shared Chebyshev coefficients.
    
    Parameters
    ----------
    band : str
        Band identifier (g, r, i)
    shared : dict
        Shared parameters including Chebyshev coefficients
    z : float
        Normalized wavelength z ∈ [-1, +1] for this band
        
    Returns
    -------
    PhysicalModel
        Physical model with wavelength-evolved parameters
    """
    # Mass model (shared across bands, using lens position and ellipticity)
    sie = SIE(
        theta_E=shared["theta_E"],
        e1=shared["e1_lens"],
        e2=shared["e2_lens"],
        center_x=shared["center_x_lens"],
        center_y=shared["center_y_lens"],
    )
    
    # Source galaxy with Chebyshev-evolved parameters
    # First create SersicEllipse with placeholder parameters
    source = SersicEllipse(
        R_sersic=1.0,  # Placeholder, will link to Chebyshev coeffs
        n_sersic=1.0,  # Placeholder, will link to Chebyshev coeffs
        e1=shared["e1_src"],
        e2=shared["e2_src"],
        center_x=shared["center_x_src"],
        center_y=shared["center_y_src"],
        Ie=ParamU(f"{band}_Ie_src", 1.0),  # Band-specific intensity (linear)
    )
    
    # Link R_sersic and n_sersic to Chebyshev coefficients using caskade
    src_R_coeffs = shared["R_sersic_src_coeffs"]
    src_n_coeffs = shared["n_sersic_src_coeffs"]
    
    # R_sersic(z) = c0*T0(z) + c1*T1(z) + c2*T2(z)
    source.R_sersic = lambda p: (
        src_R_coeffs[0].value + 
        src_R_coeffs[1].value * z + 
        src_R_coeffs[2].value * (2 * z**2 - 1)
    )
    source.R_sersic.link(src_R_coeffs)
    
    source.n_sersic = lambda p: (
        src_n_coeffs[0].value + 
        src_n_coeffs[1].value * z + 
        src_n_coeffs[2].value * (2 * z**2 - 1)
    )
    source.n_sersic.link(src_n_coeffs)
    
    # Lens galaxy with Chebyshev-evolved parameters
    lens = SersicEllipse(
        R_sersic=1.0,  # Placeholder
        n_sersic=4.0,  # Placeholder
        e1=shared["e1_lens"],
        e2=shared["e2_lens"],
        center_x=shared["center_x_lens"],
        center_y=shared["center_y_lens"],
        Ie=ParamU(f"{band}_Ie_lens", 1.0),  # Band-specific intensity (linear)
    )
    
    lens_R_coeffs = shared["R_sersic_lens_coeffs"]
    lens_n_coeffs = shared["n_sersic_lens_coeffs"]
    
    lens.R_sersic = lambda p: (
        lens_R_coeffs[0].value + 
        lens_R_coeffs[1].value * z + 
        lens_R_coeffs[2].value * (2 * z**2 - 1)
    )
    lens.R_sersic.link(lens_R_coeffs)
    
    lens.n_sersic = lambda p: (
        lens_n_coeffs[0].value + 
        lens_n_coeffs[1].value * z + 
        lens_n_coeffs[2].value * (2 * z**2 - 1)
    )
    lens.n_sersic.link(lens_n_coeffs)
    
    return PhysicalModel(lens_mass=[sie], source_light=[source], lens_light=[lens])


def weighted_quantiles(samples: np.ndarray, weights: np.ndarray) -> tuple[list[float], list[float], list[float]]:
    q16_list: list[float] = []
    q50_list: list[float] = []
    q84_list: list[float] = []
    for idx in range(samples.shape[1]):
        sorted_idx = np.argsort(samples[:, idx])
        sorted_samples = samples[sorted_idx, idx]
        sorted_weights = weights[sorted_idx]
        cumsum = np.cumsum(sorted_weights)
        cumsum /= cumsum[-1]

        q16 = float(np.interp(0.16, cumsum, sorted_samples))
        q50 = float(np.interp(0.50, cumsum, sorted_samples))
        q84 = float(np.interp(0.84, cumsum, sorted_samples))

        q16_list.append(q16)
        q50_list.append(q50)
        q84_list.append(q84)
    return q16_list, q50_list, q84_list


def plot_multiband_overview(model: MultiBandImageProbModel, theta: list[float], save_path: Path) -> None:
    model.set_values(theta)

    fig, axes = plt.subplots(3, 3, figsize=(14, 12), constrained_layout=True)
    fig.suptitle("Multi-band Lens+Source Fit with Chebyshev Evolution (g/r/i)", fontsize=16)

    for row, (band_name, band_model) in enumerate(zip(model.band_names, model.band_models)):
        forward_kwargs = {
            "use_linear": band_model.use_linear,
            "return_intensity": True,
            "ret_each_plane": True,
            "image_map": band_model.image_data,
            "noise_map": band_model.noise_map,
        }
        if not model._band_identity_geometry[row]:
            xgrid_sub, ygrid_sub = model._build_transformed_subgrid_1d(row, band_model)
            forward_kwargs["xgrid_sub"] = xgrid_sub
            forward_kwargs["ygrid_sub"] = ygrid_sub

        fwd_result = band_model.forward_model(
            **forward_kwargs,
        )
        if len(fwd_result) == 3:
            lensed_image_model, lens_light_model, _ = fwd_result
        else:
            lensed_image_model, lens_light_model = fwd_result

        data = np.asarray(band_model.image_data)
        noise = np.asarray(band_model.noise_map)
        model_image = np.asarray(lensed_image_model) + np.asarray(lens_light_model)
        residual = (data - model_image) / noise

        sim_cfg = band_model.sim_obj.sim_config
        extent = [
            -sim_cfg.npix * sim_cfg.dpix / 2.0,
            sim_cfg.npix * sim_cfg.dpix / 2.0,
            -sim_cfg.npix * sim_cfg.dpix / 2.0,
            sim_cfg.npix * sim_cfg.dpix / 2.0,
        ]

        panels = (
            (data, f"{band_name}: data", "inferno", None, None),
            (model_image, f"{band_name}: model", "inferno", None, None),
            (residual, f"{band_name}: residual", "RdBu_r", -5.0, 5.0),
        )

        for col, (img, title, cmap, vmin, vmax) in enumerate(panels):
            ax = axes[row, col]
            im = ax.imshow(img, origin="lower", extent=extent, cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_title(title)
            ax.set_xlabel("Arcsec")
            ax.set_ylabel("Arcsec")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved 9-panel overview: {save_path}")


def create_band_geometry(band: str) -> BandObservationGeometry:
    """Create BandObservationGeometry for a given band."""
    alignment = BAND_ALIGNMENTS[band]
    is_reference = band == "g"

    if FIT_ALIGNMENT_PARAMS and not is_reference:
        shift_x = ParamU(
            f"{band}_shift_x",
            alignment["shift_x"],
            prior_type="gaussian",
            prior_settings=[alignment["shift_x"], 0.02],
            limits=[-0.1, 0.1],
        )
        shift_y = ParamU(
            f"{band}_shift_y",
            alignment["shift_y"],
            prior_type="gaussian",
            prior_settings=[alignment["shift_y"], 0.02],
            limits=[-0.1, 0.1],
        )
        rotation = ParamU(
            f"{band}_rotation",
            alignment["rotation"],
            prior_type="gaussian",
            prior_settings=[alignment["rotation"], ROTATION_PRIOR_SIGMA_DEG],
            limits=[-ROTATION_LIMIT_DEG, ROTATION_LIMIT_DEG],
        )
        shift_x.to_dynamic()
        shift_y.to_dynamic()
        rotation.to_dynamic()
    else:
        shift_x = alignment["shift_x"]
        shift_y = alignment["shift_y"]
        rotation = alignment["rotation"]

    return BandObservationGeometry(
        shift_x=shift_x,
        shift_y=shift_y,
        rotation=rotation,
        is_reference=is_reference,
    )


if __name__ == "__main__":
    print("=" * 60)
    print("Multi-band Lens + Source Model with Chebyshev Evolution")
    print(f"FIT_ALIGNMENT_PARAMS = {FIT_ALIGNMENT_PARAMS}")
    print("=" * 60)

    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir / "data"
    output_dir = base_dir / "output"

    print("\n[Stage 0] Setting up Chebyshev polynomial wavelength evolution...")
    band_wavelengths = {"g": 4770.0, "r": 6231.0, "i": 7625.0}
    lambda_min, lambda_max = compute_wavelength_range(list(band_wavelengths.values()))
    print(f"  Wavelength range: [{lambda_min:.0f}, {lambda_max:.0f}] Angstroms")
    for band in BANDS:
        z = chebyshev_node(band_wavelengths[band], lambda_min, lambda_max)
        print(f"  {band}-band ({band_wavelengths[band]:.0f}A): z = {z:.3f}")

    print("\n[Stage 1] Loading g/r/i band data with heterogeneous geometry...")
    band_data_list: list[BandImageData] = []
    for band in BANDS:
        image_data, noise_map, psf_kernel, mask = load_lens_data(
            image_path=str(data_dir / f"{band}_image.fits"),
            noise_path=str(data_dir / f"{band}_noise.fits"),
            psf_path=str(data_dir / f"{band}_psf.fits"),
        )

        geometry = create_band_geometry(band)

        if band == "g":
            dpix, nsub = 0.074, 4
        elif band == "r":
            dpix, nsub = 0.08, 4
        else:  # i band
            dpix, nsub = 0.09, 4

        band_data_list.append(
            BandImageData(
                name=band,
                image_data=image_data,
                noise_map=noise_map,
                psf_kernel=psf_kernel,
                dpix=dpix,
                nsub=nsub,
                mask=mask,
                geometry=geometry,
            )
        )
    print(f"Loaded {len(band_data_list)} bands: {', '.join(BANDS)}")

    print("\n[Stage 2] Building Chebyshev-linked multi-band physical models...")
    shared_params = build_shared_chebyshev_params(lambda_min, lambda_max)
    
    # Build per-band physical models with wavelength-dependent parameters
    phys_models = []
    for band in BANDS:
        z = chebyshev_node(band_wavelengths[band], lambda_min, lambda_max)
        phys_model = build_band_physical_model(band, shared_params, z)
        phys_models.append(phys_model)
        
        # Print evolved parameters for this band
        R_src = evaluate_chebyshev_at_z(z, [0.35, -0.05, 0.02])
        n_src = evaluate_chebyshev_at_z(z, [0.9, 0.15, -0.03])
        R_lens = evaluate_chebyshev_at_z(z, [1.05, 0.02, -0.01])
        n_lens = evaluate_chebyshev_at_z(z, [4.1, -0.1, 0.05])
        print(f"  {band}-band (z={z:.3f}): src_R={R_src:.3f}, src_n={n_src:.3f}, lens_R={R_lens:.3f}, lens_n={n_lens:.3f}")

    print("\n[Stage 3] Building multi-band likelihood model...")
    likelihood = MultiBandImageProbModel(
        bands=band_data_list,
        phys_models=phys_models,
        use_linear=True,
        solver_type="nnls",
    )

    print("\n[Stage 4] Extracting priors and likelihood...")
    prior, prior_specs = make_prior_transformation(likelihood)
    param_names = [spec.name for spec in prior_specs]
    print(f"Model has {len(param_names)} dynamic parameters")
    for spec in prior_specs:
        print(f"  {spec.name}: {spec.describe()}")
        
    # Use JAX jit and vmap via make_likelihood for efficient batch processing
    jitted_loglike = make_likelihood(likelihood, vectorized=True)

    print("\n[Stage 5] Running Nautilus sampler...")
    sampler = Sampler(
        prior,
        jitted_loglike,
        n_dim=len(param_names),
        n_live=200,
        vectorized=True,
    )
    start = time.time()
    sampler.run(verbose=True, n_eff=800)
    elapsed = time.time() - start
    print(f"Sampling completed in {elapsed:.2f} seconds")

    print("\n[Stage 6] Saving posterior products...")
    samples, log_w, _ = sampler.posterior()
    weights = np.exp(log_w - np.max(log_w))
    weights /= weights.sum()

    q16_list, q50_list, q84_list = weighted_quantiles(np.asarray(samples), np.asarray(weights))
    log_z = float(np.asarray(sampler.log_z))
    linear_medians = likelihood.get_linear_solved_params(q50_list)

    output_dir.mkdir(parents=True, exist_ok=True)
    np.savetxt(
        output_dir / "result_samples.csv",
        samples,
        delimiter=",",
        header=",".join(param_names),
    )
    with (output_dir / "result_summary.csv").open("w", encoding="utf-8") as file_obj:
        file_obj.write("parameter,median,lower,upper\n")
        for idx, name in enumerate(param_names):
            file_obj.write(f"{name},{q50_list[idx]:.6f},{q16_list[idx]:.6f},{q84_list[idx]:.6f}\n")
    with gzip.open(output_dir / "results.pkl.gz", "wb") as file_obj:
        pickle.dump(
            {
                "samples": np.asarray(samples),
                "weights": np.asarray(weights),
                "log_z": log_z,
                "param_names": param_names,
                "linear_params": linear_medians,
            },
            file_obj,
        )

    print("Posterior summary:")
    for idx, name in enumerate(param_names):
        print(f"  {name:20s} = {q50_list[idx]:.4f} ({q16_list[idx]-q50_list[idx]:+.4f}, {q84_list[idx]-q50_list[idx]:+.4f})")
    print(f"log(Z) = {log_z:.3f}")
    
    # Print recovered Chebyshev coefficients
    print("\nRecovered Chebyshev coefficients:")
    for param_type in ["R_sersic_src", "n_sersic_src", "R_sersic_lens", "n_sersic_lens"]:
        coeffs = []
        for i in range(3):
            for idx, name in enumerate(param_names):
                if name == f"{param_type}_c{i}":
                    coeffs.append(q50_list[idx])
        if coeffs:
            print(f"  {param_type}: c0={coeffs[0]:.4f}, c1={coeffs[1]:.4f}, c2={coeffs[2]:.4f}")

    print("\n[Stage 7] Generating 9-panel model overview...")
    plot_multiband_overview(likelihood, q50_list, output_dir / "model_overview.png")

    print("\nSaved outputs:")
    print(f"  {output_dir / 'result_samples.csv'}")
    print(f"  {output_dir / 'result_summary.csv'}")
    print(f"  {output_dir / 'results.pkl.gz'}")
    print(f"  {output_dir / 'model_overview.png'}")

    print("\n" + "=" * 60)
    print("Inference Complete!")
    print("=" * 60)
