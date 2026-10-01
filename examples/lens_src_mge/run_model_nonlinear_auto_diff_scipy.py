"""Fit MGE lens and source light jointly using JAX gradients and L-BFGS-B.

This keeps the SIE + shear model and the two sets of 20 fixed Gaussian widths
from run_model.py. All 40 Gaussian fluxes are dynamic and use_linear=False:
there is no NNLS or other linear solve, including during initialization.
GaussianEllipse.flux is integrated flux, not central surface brightness.
Its physical bounds are [0, infinity), so components can switch off exactly.
Together with mass and shared geometry this gives 53 fitted parameters.

The objective is maximum likelihood, without prior-density penalties or a
bijector. Geometric bounds follow the original prior support and limits.
Flux coordinates are scaled by the noise-weighted norm of each unit-flux
image at the initial geometry; this computes a normalization, not a fit.

Run from any directory after activating tinylens_gpu::

    python /path/to/run_model_nonlinear_auto_diff_scipy.py --n-starts 5

Data comes from this script's data/ directory. Results go to the separate
output_auto_diff_scipy/ directory, including mge_fluxes.csv with widths,
integrated fluxes, and central surface brightnesses. Inspect all starts: successful termination
of this local optimizer can still correspond to a poor fit.
"""

import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["JAX_ENABLE_X64"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ.setdefault("MPLBACKEND", "Agg")

import argparse
import csv
import json
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

from TinyLensGpu.Inference import ParamU
from TinyLensGpu.ObservationModel.LensImage import ImageProbModel
from TinyLensGpu.PhysicalModel import PhysicalModel, GaussianEllipse, SIE, Shear
from TinyLensGpu.utils import load_lens_data, generate_radial_basis_knots
from TinyLensGpu.visualizer import plot_model_results


def make_mge(sigmas, prefix, total_flux, center_std):
    """Create fixed widths, shared dynamic geometry, and nonnegative fluxes."""
    center_x = ParamU(
        f"center_x_{prefix}",
        0.0,
        prior_type="gaussian",
        prior_settings=[0.0, center_std],
        limits=[-3.0, 3.0],
    )
    center_y = ParamU(
        f"center_y_{prefix}",
        0.0,
        prior_type="gaussian",
        prior_settings=[0.0, center_std],
        limits=[-3.0, 3.0],
    )
    # The current geometry helper gives zero gradients at exactly zero e1/e2.
    e1 = ParamU(
        f"e1_{prefix}",
        0.05,
        prior_type="gaussian",
        prior_settings=[0.0, 0.3],
        limits=[-1.0, 1.0],
    )
    e2 = ParamU(
        f"e2_{prefix}",
        0.05,
        prior_type="gaussian",
        prior_settings=[0.0, 0.3],
        limits=[-1.0, 1.0],
    )
    for param in (center_x, center_y, e1, e2):
        param.to_dynamic()
    components = []
    for index, sigma in enumerate(sigmas):
        component = GaussianEllipse(
            sigma=ParamU(f"sigma_{prefix}_{index}", float(sigma)),
            center_x=center_x,
            center_y=center_y,
            e1=e1,
            e2=e2,
            # Bounds are for optimization; no sampling prior is needed.
            flux=ParamU(
                f"flux_{prefix}_{index}", total_flux / len(sigmas), limits=[0.0, np.inf]
            ),
        )
        component.sigma.to_static(float(sigma))
        component.flux.to_dynamic()
        components.append(component)
    return components


def build_problem(data_dir):
    """Preserve the original 20+20 MGE basis and optimize its fluxes directly."""
    image, noise, psf, mask = load_lens_data(
        image_path=str(data_dir / "image.fits"),
        noise_path=str(data_dir / "noise.fits"),
        psf_path=str(data_dir / "psf.fits"),
    )
    if not all(np.all(np.isfinite(array)) for array in (image, noise, psf)):
        raise ValueError("Image, noise, and PSF must contain only finite values")
    image, noise, psf = (array.astype(np.float64) for array in (image, noise, psf))
    sie = SIE(
        theta_E=ParamU(
            "theta_E",
            1.5,
            prior_type="uniform",
            prior_settings=[0.001, 3.001],
            limits=[0.0, 10.0],
        ),
        e1=ParamU(
            "e1_mass",
            0.05,
            prior_type="gaussian",
            prior_settings=[0.0, 0.3],
            limits=[-1.0, 1.0],
        ),
        e2=ParamU(
            "e2_mass",
            0.05,
            prior_type="gaussian",
            prior_settings=[0.0, 0.3],
            limits=[-1.0, 1.0],
        ),
        center_x=ParamU("center_x_mass", 0.0),
        center_y=ParamU("center_y_mass", 0.0),
    )
    shear = Shear(
        gamma1=ParamU(
            "gamma1",
            0.0,
            prior_type="uniform",
            prior_settings=[-0.2, 0.2],
            limits=[-0.5, 0.5],
        ),
        gamma2=ParamU(
            "gamma2",
            0.0,
            prior_type="uniform",
            prior_settings=[-0.2, 0.2],
            limits=[-0.5, 0.5],
        ),
    )
    for param in (sie.theta_E, sie.e1, sie.e2, shear.gamma1, shear.gamma2):
        param.to_dynamic()
    sie.center_x.to_static(0.0)
    sie.center_y.to_static(0.0)
    source_sigmas = generate_radial_basis_knots(
        dpix=0.074,
        n_sigmas=20,
        log_rmin=-2.0,
        log_rmax=np.log10(1.0),
        mode="mge",
    )
    lens_sigmas = generate_radial_basis_knots(
        dpix=0.074,
        center_x=0.0,
        center_y=0.0,
        n_sigmas=20,
        log_rmin=-2.0,
        log_rmax=np.log10(3.0),
        arc_mask=mask,
        mode="mge",
    )
    # Simple fixed positive guesses, with no data fitting or linear solve.
    source = make_mge(source_sigmas, "src", total_flux=1.0, center_std=0.5)
    lens = make_mge(lens_sigmas, "lens", total_flux=20.0, center_std=0.1)
    model = ImageProbModel(
        image_data=image,
        noise_map=noise,
        psf_kernel=psf,
        dpix=0.074,
        nsub=4,
        phys_model=PhysicalModel(
            lens_mass=[sie, shear], source_light=source, lens_light=lens
        ),
        use_linear=False,
        mask=mask,
    )
    params = list(model.get_dynamic_params())
    names = [param.name for param in params]
    initial = np.asarray([float(param.value) for param in params])
    bounds = []
    for param in params:
        lower, upper = param.limits
        if (
            param.prior_type in ("uniform", "log_uniform")
            and param.prior_settings is not None
        ):
            lower = max(lower, param.prior_settings[0])
            upper = min(upper, param.prior_settings[1])
        bounds.append((lower, upper))
    scales = np.asarray(
        [
            (
                0.1
                if name.startswith(("e1_", "e2_", "gamma"))
                else 0.3 if name.startswith("center_") else 1.0
            )
            for name in names
        ]
    )
    return model, names, initial, np.asarray(bounds), scales


def scale_flux_coordinates(model, names, initial, scales):
    """Normalize each flux by its unit-image response, without solving for it."""
    scales = scales.copy()
    flux_indices = [i for i, name in enumerate(names) if name.startswith("flux_")]
    reference = initial.copy()
    reference[flux_indices] = 0.0
    n_pixels = int(np.asarray(model.unmask).sum())
    noise = np.asarray(model.noise_map)
    unmask = np.asarray(model.unmask, dtype=bool)
    forward = jax.jit(lambda theta: model.forward_model(theta, return_intensity=False))
    for index in flux_indices:
        theta = reference.copy()
        theta[index] = 1.0
        template = np.asarray(forward(jnp.asarray(theta, dtype=jnp.float64)))
        norm = np.linalg.norm((template / noise)[unmask])
        if not np.isfinite(norm) or norm <= np.finfo(float).eps:
            raise ValueError(f"No finite unit-flux response for {names[index]}")
        scales[index] = np.sqrt(n_pixels) / norm
    return scales


def make_differentiable_loglike(likelihood):
    """Reuse the model and likelihood math with theta explicit at the JIT boundary.

    ImageProbModel.__call__ nests a JIT with only self as an argument inside
    Caskade's parameter injection. Under autodiff that JIT can cache captured
    parameter tracers. Calling the forward model and the shared likelihood
    evaluator lets our outer JIT see theta as an ordinary argument instead.
    """
    if likelihood.use_linear:
        raise ValueError("This example requires use_linear=False")

    def loglike(theta):
        image_model, intensities = likelihood.forward_model(theta)
        return likelihood._evaluate_loglike_from_forward_model(image_model, intensities)

    return jax.jit(loglike)


def make_objective(likelihood, scales):
    """Keep tracing in JAX; convert to Python/NumPy only at the SciPy boundary."""
    n_pixels = int(np.asarray(likelihood.unmask).sum())
    if n_pixels <= len(scales):
        raise ValueError("There must be more unmasked pixels than fitted parameters")
    scales_jax = jnp.asarray(scales, dtype=jnp.float64)
    loglike = make_differentiable_loglike(likelihood)

    def loss(x):
        theta = x * scales_jax
        # Use a JAX scalar interface. make_likelihood's sampler
        # wrapper converts single evaluations to float and cannot be traced.
        # Remove the constant and normalize by pixel count: chi2 / (2*N).
        # Both operations preserve the maximum-likelihood solution.
        return (-loglike(theta) + likelihood.log_like_const) / n_pixels

    value_and_grad = jax.jit(jax.value_and_grad(loss))

    def scipy_objective(x):
        value, grad = value_and_grad(jnp.asarray(x, dtype=jnp.float64))
        value, grad = float(value), np.asarray(grad, dtype=np.float64)
        if not np.isfinite(value) or not np.all(np.isfinite(grad)):
            raise FloatingPointError(
                "Nonfinite likelihood or gradient during optimization"
            )
        return value, grad

    return scipy_objective, n_pixels


def check_gradient(objective, x, names, scales, stage):
    """Check all autodiff components against central finite differences."""
    _, autodiff = objective(x)
    rows = []
    for index, name in enumerate(names):
        step = 1e-5 * max(1.0, abs(x[index]))
        offset = np.zeros_like(x)
        offset[index] = step
        finite_difference = (objective(x + offset)[0] - objective(x - offset)[0]) / (
            2 * step
        )
        # At the optimum the derivative may vanish, so use an absolute as well
        # as a relative tolerance, in the scaled optimizer coordinates.
        passed = bool(
            np.isclose(autodiff[index], finite_difference, rtol=2e-5, atol=1e-7)
        )
        rows.append(
            {
                "stage": stage,
                "parameter": name,
                "autodiff_physical": autodiff[index] / scales[index],
                "finite_difference_physical": finite_difference / scales[index],
                "passed": passed,
            }
        )
        if not passed:
            raise RuntimeError(
                f"Gradient check failed for {name} at {stage}: {rows[-1]}"
            )
    print(f"Gradient check ({stage}): all {len(names)} parameters passed")
    return rows


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=script_dir / "data")
    parser.add_argument(
        "--output-dir", type=Path, default=script_dir / "output_auto_diff_scipy"
    )
    parser.add_argument("--n-starts", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--maxiter", type=int, default=3000)
    args = parser.parse_args()
    if args.n_starts < 1 or args.maxiter < 1:
        parser.error("--n-starts and --maxiter must be positive")

    print("Nonlinear MGE lens + source maximum likelihood: JAX + SciPy L-BFGS-B")
    print(f"JAX devices: {jax.devices()}; float64: {jax.config.x64_enabled}")
    likelihood, names, initial, bounds, scales = build_problem(args.data_dir)
    scaling_start = time.perf_counter()
    print("Normalizing the 40 flux coordinates using unit-flux images...")
    scales = scale_flux_coordinates(likelihood, names, initial, scales)
    scaling_seconds = time.perf_counter() - scaling_start
    print(f"Flux normalization: {scaling_seconds:.2f} s")
    objective, n_pixels = make_objective(likelihood, scales)
    x0 = initial / scales
    scaled_bounds = bounds / scales[:, None]
    print(f"Dynamic order: {names}; unmasked pixels: {n_pixels}")
    print(f"Physical bounds: {dict(zip(names, bounds.tolist()))}")

    warmup_start = time.perf_counter()
    initial_loss, _ = objective(x0)
    warmup_seconds = time.perf_counter() - warmup_start
    print(
        f"JIT warmup: {warmup_seconds:.2f} s; initial chi2: {2 * n_pixels * initial_loss:.6f}"
    )
    gradient_checks = check_gradient(objective, x0, names, scales, "initial")

    rng = np.random.default_rng(args.seed)
    starts = [initial]
    for _ in range(args.n_starts - 1):
        # Independent, reproducible starting points; no posterior or mock
        # truth values are used to initialize the optimization.
        values = dict(zip(names, initial))
        values.update(
            {
                "theta_E": rng.uniform(1.0, 2.0),
                "e1_mass": rng.uniform(-0.15, 0.15),
                "e2_mass": rng.uniform(-0.15, 0.15),
                "gamma1": rng.uniform(-0.1, 0.1),
                "gamma2": rng.uniform(-0.1, 0.1),
                "e1_src": rng.uniform(-0.2, 0.2),
                "e2_src": rng.uniform(-0.2, 0.2),
                "center_x_src": rng.uniform(-0.4, 0.4),
                "center_y_src": rng.uniform(-0.4, 0.4),
                "e1_lens": rng.uniform(-0.2, 0.2),
                "e2_lens": rng.uniform(-0.2, 0.2),
                "center_x_lens": rng.uniform(-0.02, 0.02),
                "center_y_lens": rng.uniform(-0.02, 0.02),
            }
        )
        for name in names:
            if name.startswith("flux_"):
                values[name] *= np.exp(rng.uniform(-0.7, 0.7))
        starts.append(np.asarray([values[name] for name in names]))

    options = {
        "maxiter": args.maxiter,
        "maxls": 50,
        "maxcor": 50,
        "ftol": 1e-12,
        "gtol": 1e-8,
    }
    results, run_rows, history = [], [], []
    optimization_start = time.perf_counter()
    for start_id, theta0 in enumerate(starts):
        start_time = time.perf_counter()
        iteration = 0

        def record(x):
            nonlocal iteration
            loss, _ = objective(x)
            history.append(
                {
                    "start": start_id,
                    "iteration": iteration,
                    "chi2": 2 * n_pixels * loss,
                    **dict(zip(names, x * scales)),
                }
            )
            if iteration > 0 and iteration % 25 == 0:
                print(
                    f"  Start {start_id + 1}, iteration {iteration}: chi2={2 * n_pixels * loss:.6f}"
                )
            iteration += 1

        record(theta0 / scales)
        result = minimize(
            objective,
            theta0 / scales,
            method="L-BFGS-B",
            jac=True,
            bounds=scaled_bounds,
            options=options,
            callback=record,
        )
        results.append(result)
        theta = result.x * scales
        fitted = dict(zip(names, theta))
        # The original component clips ellipticity norm at 0.9999. Do not
        # accept a stationary result in that saturated, nonphysical region.
        valid_ellipticity = all(
            np.hypot(fitted[f"e1_{component}"], fitted[f"e2_{component}"]) < 0.9999
            for component in ("mass", "src", "lens")
        )
        run_rows.append(
            {
                "start": start_id,
                "success": bool(result.success),
                "message": str(result.message),
                "valid_ellipticity": valid_ellipticity,
                "chi2": 2 * n_pixels * result.fun,
                "iterations": int(result.nit),
                "evaluations": int(result.nfev),
                "seconds": time.perf_counter() - start_time,
                **{f"initial_{name}": value for name, value in zip(names, theta0)},
                **fitted,
            }
        )
        print(
            f"Start {start_id + 1}/{args.n_starts}: success={result.success}, "
            f"chi2={run_rows[-1]['chi2']:.6f}, iterations={result.nit}, "
            f"time={run_rows[-1]['seconds']:.2f} s; {result.message}"
        )
    optimization_seconds = time.perf_counter() - optimization_start

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "optimization_runs.csv", run_rows)
    write_csv(output_dir / "optimization_history.csv", history)
    valid_indices = [
        i
        for i, row in enumerate(run_rows)
        if row["valid_ellipticity"] and np.isfinite(results[i].fun)
    ]
    if not valid_indices:
        raise RuntimeError(
            f"No physical best fit found; inspect {output_dir / 'optimization_runs.csv'}"
        )
    best_id = min(valid_indices, key=lambda i: results[i].fun)
    best = results[best_id]
    theta_best = best.x * scales
    gradient_checks += check_gradient(objective, best.x, names, scales, "best_fit")
    chi2 = float(2 * n_pixels * best.fun)
    matching_chi2_tolerance = 1e-3
    starts_matching_best = sum(
        row["valid_ellipticity"] and abs(row["chi2"] - chi2) <= matching_chi2_tolerance
        for row in run_rows
    )
    loglike = float(make_differentiable_loglike(likelihood)(jnp.asarray(theta_best)))
    # Independently verify the objective against the image residuals.
    model_image = np.asarray(
        likelihood.forward_model(jnp.asarray(theta_best), return_intensity=False)
    )
    residuals = (model_image - np.asarray(likelihood.image_data)) / np.asarray(
        likelihood.noise_map
    )
    chi2_from_image = float(
        np.sum(residuals[np.asarray(likelihood.unmask, dtype=bool)] ** 2)
    )
    if not np.isclose(chi2, chi2_from_image, rtol=1e-10, atol=1e-8):
        raise RuntimeError(
            "Optimization objective does not match the image residual chi2"
        )

    projected_gradient = best.jac.copy()
    projected_gradient[(best.x <= scaled_bounds[:, 0]) & (best.jac > 0)] = 0
    projected_gradient[(best.x >= scaled_bounds[:, 1]) & (best.jac < 0)] = 0
    fitted = dict(zip(names, theta_best))
    mge_rows = []
    for prefix, components in (
        ("src", likelihood.phys_model.source_light),
        ("lens", likelihood.phys_model.lens_light),
    ):
        for index, component in enumerate(components):
            sigma = float(component.sigma.value)
            flux = float(fitted[f"flux_{prefix}_{index}"])
            mge_rows.append(
                {
                    "plane": prefix,
                    "component": index,
                    "sigma": sigma,
                    "flux": flux,
                    "central_surface_brightness": flux / (2 * np.pi * sigma**2),
                    "at_zero_flux": flux == 0.0,
                }
            )
    write_csv(output_dir / "mge_fluxes.csv", mge_rows)
    summary = {
        "method": "L-BFGS-B",
        "objective": "maximum likelihood; chi2 / (2 * unmasked_pixels)",
        "model": "SIE + Shear; 20 source and 20 lens Gaussian components",
        "use_linear": False,
        "dpix": 0.074,
        "nsub": 4,
        "data_dir": str(args.data_dir.resolve()),
        "devices": [str(device) for device in jax.devices()],
        "seed": args.seed,
        "options": options,
        "parameter_order": names,
        "physical_bounds": {
            name: [float(value) if np.isfinite(value) else None for value in pair]
            for name, pair in zip(names, bounds)
        },
        "scales": dict(zip(names, scales.tolist())),
        "flux_scaling": "sqrt(unmasked_pixels) / noise-weighted unit-image norm at initial geometry",
        "flux_scaling_seconds": scaling_seconds,
        "source_sigmas": [
            float(component.sigma.value)
            for component in likelihood.phys_model.source_light
        ],
        "lens_sigmas": [
            float(component.sigma.value)
            for component in likelihood.phys_model.lens_light
        ],
        "nonnegative_fluxes": True,
        "linear_solve_used": False,
        "zero_flux_components": sum(row["at_zero_flux"] for row in mge_rows),
        "total_source_flux": sum(
            row["flux"] for row in mge_rows if row["plane"] == "src"
        ),
        "total_lens_flux": sum(
            row["flux"] for row in mge_rows if row["plane"] == "lens"
        ),
        "best_start": best_id,
        "success": bool(best.success),
        "message": str(best.message),
        "best_fit": dict(zip(names, theta_best.tolist())),
        "log_likelihood": loglike,
        "initial_chi2": float(2 * n_pixels * initial_loss),
        "chi2": chi2,
        "unmasked_pixels": n_pixels,
        "degrees_of_freedom": n_pixels - len(names),
        "reduced_chi2": chi2 / (n_pixels - len(names)),
        "projected_gradient_max_abs": float(np.max(np.abs(projected_gradient))),
        "jit_warmup_seconds": warmup_seconds,
        "optimization_seconds": optimization_seconds,
        "n_starts": args.n_starts,
        "successful_starts": sum(row["success"] for row in run_rows),
        "starts_matching_best_chi2": starts_matching_best,
        "matching_chi2_absolute_tolerance": matching_chi2_tolerance,
        "gradient_checks_passed": all(row["passed"] for row in gradient_checks),
    }
    write_csv(output_dir / "gradient_check.csv", gradient_checks)
    write_csv(
        output_dir / "best_fit.csv",
        [
            {
                "parameter": name,
                "value": value,
                "lower_bound": lower,
                "upper_bound": upper if np.isfinite(upper) else "",
            }
            for name, value, (lower, upper) in zip(names, theta_best, bounds)
        ],
    )
    with (output_dir / "fit_summary.json").open("w") as stream:
        json.dump(summary, stream, indent=2)
        stream.write("\n")
    figure, _ = plot_model_results(
        likelihood,
        theta_best,
        save_path=str(output_dir / "model_visualization.png"),
        title=f"JAX + SciPy L-BFGS-B: reduced chi2 = {summary['reduced_chi2']:.4f}",
        show_critical_lines=True,
        show_caustics=True,
    )
    plt.close(figure)

    print("\nBest-fit physical parameters:")
    for name, value in zip(names, theta_best):
        print(f"  {name:12s} = {value:.10g}")
    print(f"log L = {loglike:.6f}; reduced chi2 = {summary['reduced_chi2']:.6f}")
    print(f"Starts matching best chi2: {starts_matching_best}/{args.n_starts}")
    print(
        f"Optimization: {optimization_seconds:.2f} s (excluding JIT and gradient checks)"
    )
    print(f"Results saved to {output_dir}")
    return 0 if best.success else 1


if __name__ == "__main__":
    raise SystemExit(main())
