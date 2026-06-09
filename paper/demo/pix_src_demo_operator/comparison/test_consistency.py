"""Consistency test: matrix vs operator backends for pixelized source inversion.

Compares source reconstruction, model image, chi-squared, and log evidence
between the two backends across a range of regularization strengths.
"""

import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

from pathlib import Path
os.chdir(Path(__file__).parent.parent / "simple" / "pix_src")

import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt

from TinyLensGpu.PhysicalModel import PhysicalModel, SIE
from TinyLensGpu.PhysicalModel.LensImage.Pixelized.Light import PixelizedSourceModel
from TinyLensGpu.ObservationModel.LensImage.pixelized_image_model import (
    PixelizedImageProbModel,
)
from TinyLensGpu.ObservationModel.LensImage.pixelized_image_model_operator import (
    PixelizedImageProbModelOperator,
)
from TinyLensGpu.utils import load_lens_data

# ------------------------------------------------------------------ #
# Data
# ------------------------------------------------------------------ #
image_data, noise_map, psf_kernel, mask = load_lens_data(
    image_path="data/image.fits",
    noise_path="data/noise.fits",
    psf_path="data/psf.fits",
    mask_path="data/mask.fits",
)

sie = SIE(theta_E=1.0, e1=0.1, e2=0.0, center_x=0.0, center_y=0.0)
DPIX = 0.05

LAMBDA_LIST = [1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100]

print(f"{'λ':>10s} | {'src RMS rel':>12s} | {'model RMS rel':>12s} | {'chi2 diff':>10s} | {'log_ev diff':>12s}")
print("-" * 75)

results = []

for LAMBDA in LAMBDA_LIST:
    pix_src = PixelizedSourceModel(
        nx=40, ny=40,
        regularization_type="first-order",
        lambda_reg=LAMBDA,
    )
    phys = PhysicalModel(lens_mass=[sie], source_light=[pix_src], lens_light=[])

    # Matrix backend
    pm_mat = PixelizedImageProbModel(image_data, noise_map, psf_kernel, DPIX,
                                     phys, mask=mask)
    _, src_mat = pm_mat.forward_model(return_source=True)
    log_ev_mat = pm_mat.likelihood()
    model_mat = pm_mat.forward_model()
    chi2_mat = float(jnp.sum(((jnp.asarray(image_data) - model_mat)[~mask] /
                               jnp.asarray(noise_map)[~mask])**2))

    # Operator backend
    pm_op = PixelizedImageProbModelOperator(image_data, noise_map, psf_kernel, DPIX,
                                            phys, mask=mask)
    _, src_op = pm_op.forward_model(return_source=True)
    log_ev_op = pm_op.likelihood()
    model_op = pm_op.forward_model()
    chi2_op = float(jnp.sum(((jnp.asarray(image_data) - model_op)[~mask] /
                              jnp.asarray(noise_map)[~mask])**2))

    src_rms_rel = float(jnp.sqrt(jnp.mean((src_mat - src_op)**2)) /
                        jnp.sqrt(jnp.mean(src_mat**2)))
    model_rms_rel = float(jnp.sqrt(jnp.mean((model_mat - model_op)**2)) /
                          jnp.sqrt(jnp.mean(model_mat**2)))
    chi2_diff = float(chi2_mat - chi2_op)
    lev_diff = float(log_ev_mat - log_ev_op)

    print(f"{LAMBDA:10.1e} | {src_rms_rel:12.2e} | {model_rms_rel:12.2e} | "
          f"{chi2_diff:10.4f} | {lev_diff:12.4f}")

    results.append({
        'lambda': LAMBDA,
        'src_rms_rel': src_rms_rel,
        'model_rms_rel': model_rms_rel,
        'chi2_diff': chi2_diff,
        'log_ev_diff': lev_diff,
    })

# Summary
print()
print("=" * 75)
print("Consistency check complete.")
print(f"  Source RMS relative error:     {np.mean([r['src_rms_rel'] for r in results]):.2e}")
print(f"  Model image RMS relative error: {np.mean([r['model_rms_rel'] for r in results]):.2e}")
print(f"  Mean chi2 difference:           {np.mean([r['chi2_diff'] for r in results]):.4f}")
print(f"  Mean log_ev difference:         {np.mean([r['log_ev_diff'] for r in results]):.4f}")
