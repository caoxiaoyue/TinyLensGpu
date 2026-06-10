# pyright: reportMissingImports=false

from __future__ import annotations

import gzip
import os
import pickle
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from nautilus import Sampler

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

from TinyLensGpu.Inference import ParamU, nautilus_posterior_summary
from TinyLensGpu.Inference.build_likelihood import make_likelihood
from TinyLensGpu.Inference.build_prior import make_prior_transformation
from TinyLensGpu.ObservationModel import BandImageData, MultiBandImageProbModel
from TinyLensGpu.ObservationModel.LensImage.multi_band_image_model import BandObservationGeometry
from TinyLensGpu.PhysicalModel.LensImage.Parametric.Light import SersicEllipse
from TinyLensGpu.PhysicalModel.LensImage.Parametric.Mass import SIE
from TinyLensGpu.PhysicalModel.LensImage.composite import PhysicalModel
from TinyLensGpu.utils import load_lens_data


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


def build_shared_params() -> dict[str, ParamU]:
    return {
        "theta_E": ParamU("theta_E", 1.5, prior_type="uniform", prior_settings=[0.001, 3.001], limits=[0.0, 10.0]),
        "e1_mass": ParamU("e1_mass", 0.0, prior_type="gaussian", prior_settings=[0.0, 0.3], limits=[-1.0, 1.0]),
        "e2_mass": ParamU("e2_mass", 0.0, prior_type="gaussian", prior_settings=[0.0, 0.3], limits=[-1.0, 1.0]),
        "center_x_mass": ParamU("center_x_mass", 0.0),
        "center_y_mass": ParamU("center_y_mass", 0.0),
        "R_sersic_src": ParamU("R_sersic_src", 0.8, prior_type="uniform", prior_settings=[0.001, 2.001], limits=[0.0, 5.0]),
        "n_sersic_src": ParamU("n_sersic_src", 1.0, prior_type="uniform", prior_settings=[0.3, 2.3], limits=[0.3, 6.0]),
        "e1_src": ParamU("e1_src", 0.0, prior_type="gaussian", prior_settings=[0.0, 0.3], limits=[-1.0, 1.0]),
        "e2_src": ParamU("e2_src", 0.0, prior_type="gaussian", prior_settings=[0.0, 0.3], limits=[-1.0, 1.0]),
        "center_x_src": ParamU("center_x_src", 0.0, prior_type="gaussian", prior_settings=[0.0, 0.5], limits=[-3.0, 3.0]),
        "center_y_src": ParamU("center_y_src", 0.3, prior_type="gaussian", prior_settings=[0.3, 0.5], limits=[-3.0, 3.0]),
        "R_sersic_lens": ParamU("R_sersic_lens", 1.0, prior_type="uniform", prior_settings=[0.001, 2.001], limits=[0.0, 5.0]),
        "n_sersic_lens": ParamU("n_sersic_lens", 4.0, prior_type="gaussian", prior_settings=[4.0, 0.5], limits=[0.3, 6.0]),
        "e1_lens": ParamU("e1_lens", 0.0, prior_type="gaussian", prior_settings=[0.0, 0.3], limits=[-1.0, 1.0]),
        "e2_lens": ParamU("e2_lens", 0.0, prior_type="gaussian", prior_settings=[0.0, 0.3], limits=[-1.0, 1.0]),
        "center_x_lens": ParamU("center_x_lens", 0.0),
        "center_y_lens": ParamU("center_y_lens", 0.0),
    }


def build_band_physical_model(band: str, shared: dict[str, ParamU]) -> PhysicalModel:
    sie = SIE(
        theta_E=shared["theta_E"],
        e1=shared["e1_mass"],
        e2=shared["e2_mass"],
        center_x=shared["center_x_mass"],
        center_y=shared["center_y_mass"],
    )

    source = SersicEllipse(
        R_sersic=shared["R_sersic_src"],
        n_sersic=shared["n_sersic_src"],
        e1=shared["e1_src"],
        e2=shared["e2_src"],
        center_x=shared["center_x_src"],
        center_y=shared["center_y_src"],
        Ie=ParamU(f"{band}_Ie_src", 1.0),
    )

    lens = SersicEllipse(
        R_sersic=shared["R_sersic_lens"],
        n_sersic=shared["n_sersic_lens"],
        e1=shared["e1_lens"],
        e2=shared["e2_lens"],
        center_x=shared["center_x_lens"],
        center_y=shared["center_y_lens"],
        Ie=ParamU(f"{band}_Ie_lens", 1.0),
    )

    return PhysicalModel(lens_mass=[sie], source_light=[source], lens_light=[lens])


def set_dynamic_params(shared: dict[str, ParamU]) -> None:
    dynamic_names = (
        "theta_E",
        "e1_mass",
        "e2_mass",
        "R_sersic_src",
        "n_sersic_src",
        "e1_src",
        "e2_src",
        "center_x_src",
        "center_y_src",
        "R_sersic_lens",
        "n_sersic_lens",
        "e1_lens",
        "e2_lens",
    )
    for name in dynamic_names:
        shared[name].to_dynamic()


def plot_multiband_overview(model: MultiBandImageProbModel, theta: list[float], save_path: Path) -> None:
    model.set_values(theta)

    fig, axes = plt.subplots(3, 3, figsize=(14, 12), constrained_layout=True)
    fig.suptitle("Multi-band Lens+Source Fit (g/r/i)", fontsize=16)

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

    # Overlay critical lines on data and model panels
    from TinyLensGpu.visualizer import overlay_critical_lines
    ref_model = model.band_models[0]
    lens_mass = ref_model.sim_obj.phys_model.lens_mass
    for row in range(3):
        for col in (0, 1):  # data and model columns only
            overlay_critical_lines(axes[row, col], lens_mass, x_range=(-3.0, 3.0), y_range=(-3.0, 3.0))

    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved 9-panel overview: {save_path}")


def create_band_geometry(band: str) -> BandObservationGeometry:
    """Create BandObservationGeometry for a given band.

    g band is the reference band (is_reference=True).
    For other bands, optionally promote alignment params to dynamic ParamUs
    when FIT_ALIGNMENT_PARAMS is True.
    """
    alignment = BAND_ALIGNMENTS[band]
    is_reference = band == "g"

    if FIT_ALIGNMENT_PARAMS and not is_reference:
        # Promote alignment params to dynamic ParamUs with explicit priors
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
        # Mark as dynamic
        shift_x.to_dynamic()
        shift_y.to_dynamic()
        rotation.to_dynamic()
    else:
        # Use static values (reference band always uses static)
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
    print("Multi-band Lens + Source Model Inference")
    print(f"FIT_ALIGNMENT_PARAMS = {FIT_ALIGNMENT_PARAMS}")
    print("=" * 60)

    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir / "data"
    output_dir = base_dir / "output"

    print("\n[Stage 1] Loading g/r/i band data with heterogeneous geometry...")
    band_data_list: list[BandImageData] = []
    for band in BANDS:
        image_data, noise_map, psf_kernel, mask = load_lens_data(
            image_path=str(data_dir / f"{band}_image.fits"),
            noise_path=str(data_dir / f"{band}_noise.fits"),
            psf_path=str(data_dir / f"{band}_psf.fits"),
        )

        # Get per-band geometry configuration
        geometry = create_band_geometry(band)

        # Determine dpix and nsub based on the heterogeneous setup from sim_data.py
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

    print("\n[Stage 2] Building tied multi-band physical models...")
    shared_params = build_shared_params()
    set_dynamic_params(shared_params)
    phys_models = [build_band_physical_model(band, shared_params) for band in BANDS]

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
    loglike = make_likelihood(likelihood, vectorized=True)

    print("\n[Stage 5] Running Nautilus sampler...")
    sampler = Sampler(
        prior,
        loglike,
        n_dim=len(param_names),
        n_live=200,
        vectorized=True,
        n_batch=100,
    )
    start = time.time()
    sampler.run(verbose=True, n_eff=800)
    elapsed = time.time() - start
    print(f"Sampling completed in {elapsed:.2f} seconds")

    print("\n[Stage 6] Saving posterior products...")
    samples, weights, quantiles, log_z = nautilus_posterior_summary(sampler, param_names)
    q16_list = [float(qs[0]) for qs in quantiles.values()]
    q50_list = [float(qs[1]) for qs in quantiles.values()]
    q84_list = [float(qs[2]) for qs in quantiles.values()]
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
