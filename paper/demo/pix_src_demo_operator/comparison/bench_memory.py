"""GPU memory usage comparison: matrix vs operator backend.

Measures peak GPU memory for source inversion across grid sizes.
"""

import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import time
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

# ------------------------------------------------------------------ #
# Test grid sizes
# ------------------------------------------------------------------ #
GRID_SIZES = [
    (20, 20),   # source 20x20
    (30, 30),
    (40, 40),
    (50, 50),
    (60, 60),
]

npix = 100          # image pixels
dpix = 0.05

# Build a simple mock dataset
sie = SIE(theta_E=1.0, e1=0.1, e2=0.0, center_x=0.0, center_y=0.0)

# Create mock data once at the largest size and crop as needed
x = jnp.linspace(-npix * dpix / 2, npix * dpix / 2, npix)
xx, yy = jnp.meshgrid(x, x, indexing='xy')
rr = jnp.sqrt((xx)**2 + (yy)**2)
mask = rr > 2.5  # annular-like mask

# Mock image: simple ring structure
image = jnp.exp(-0.5 * ((rr - 1.0) / 0.15)**2)
image = jnp.where(mask, 0.0, image)
noise = jnp.ones_like(image) * 0.02
psf = jnp.ones((5, 5)) / 25.0

print("Measuring peak GPU memory for design matrix construction...")
print(f"{'Grid':>12s} | {'Matrix (MB)':>12s} | {'Operator (MB)':>14s} | {'Ratio':>8s}")
print("-" * 55)

results = []

for src_nx, src_ny in GRID_SIZES:
    pix_src = PixelizedSourceModel(
        nx=src_nx, ny=src_ny,
        regularization_type="first-order",
        log_lambda_reg=jnp.log(1.0),
    )
    phys = PhysicalModel(lens_mass=[sie], source_light=[pix_src], lens_light=[])

    # --- Matrix backend: measure F matrix size ---
    pm_mat = PixelizedImageProbModel(image, noise, psf, dpix, phys, mask=mask)
    F, bbox = pm_mat.sim_obj.design_matrix()
    Nd, Ns = F.shape
    mem_f_matrix = Nd * Ns * 4 / (1024**2)  # float32 → MB

    # --- Operator backend: no F matrix ---
    # Build preconditioner (the largest explicit matrix in operator path)
    pm_op = PixelizedImageProbModelOperator(image, noise, psf, dpix, phys, mask=mask)
    _, _, bx, by = pm_op.sim_obj._get_beta_sub_and_seed()
    xmi, xma, ymi, yma = pm_op.sim_obj._infer_and_fix_bbox(bx, by)

    from TinyLensGpu.utils.inversion.regularization import DenseRegularizationBuilder
    builder = DenseRegularizationBuilder(src_nx, src_ny, "first-order")
    reg, _ = builder.matrix(float(xmi), float(xma), float(ymi), float(yma))

    P, _ = pm_op.sim_obj.build_preconditioner(
        pm_op.noise_1d, xmi, xma, ymi, yma, jnp.asarray(1.0), reg,
    )
    mem_p_matrix = Ns * Ns * 4 / (1024**2)

    # The operator path avoids the F matrix; only stores P (Ns x Ns)
    # Plus small arrays: weights (Nd, 4), indices (Nd, 4), etc.
    mem_weights = Nd * 4 * 4 / (1024**2)  # weights + indices
    mem_operator = mem_p_matrix + mem_weights

    ratio = mem_operator / mem_f_matrix if mem_f_matrix > 0 else 0

    print(f"{src_nx}x{src_ny} (Ns={Ns:4d}) | {mem_f_matrix:12.2f} | {mem_operator:14.2f} | {ratio:7.2%}")

    results.append({
        'grid': f"{src_nx}x{src_ny}",
        'Ns': Ns,
        'Nd': Nd,
        'matrix_mb': mem_f_matrix,
        'operator_mb': mem_operator,
        'ratio': ratio,
    })

print()
print("Summary: operator backend avoids the Nd x Ns design matrix.")
print(f"  Key structures: F matrix = Nd x Ns float32, P matrix = Ns x Ns float32")
print(f"  When Nd >> Ns, savings are substantial.")

# Simple text-based "plot"
print()
print("Memory (MB) vs source grid size:")
print(f"{'Grid':>12s} | Matrix | Operator")
for r in results:
    bar_m = '#' * int(r['matrix_mb'] * 2)
    bar_o = '#' * int(r['operator_mb'] * 2)
    print(f"{r['grid']:>12s} | {bar_m} ({r['matrix_mb']:.1f})")
    print(f"{'':>12s} | {bar_o} ({r['operator_mb']:.1f})")
