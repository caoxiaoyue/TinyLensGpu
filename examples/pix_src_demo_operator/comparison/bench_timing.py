"""Computation time comparison: matrix vs operator backends.

Measures wall-clock time for source inversion and evidence evaluation.
"""

import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import time
import numpy as np
import jax.numpy as jnp

from TinyLensGpu.PhysicalModel import PhysicalModel, SIE
from TinyLensGpu.PhysicalModel.LensImage.Pixelized.Light import PixelizedSourceModel
from TinyLensGpu.ObservationModel.LensImage.pixelized_image_model import (
    PixelizedImageProbModel,
)
from TinyLensGpu.ObservationModel.LensImage.pixelized_image_model_operator import (
    PixelizedImageProbModelOperator,
)

# ------------------------------------------------------------------ #
# Test parameters
# ------------------------------------------------------------------ #
npix = 100
dpix = 0.05
N_WARMUP = 2
N_REPEAT = 5

# Mock data
x = jnp.linspace(-npix * dpix / 2, npix * dpix / 2, npix)
xx, yy = jnp.meshgrid(x, x, indexing='xy')
rr = jnp.sqrt(xx**2 + yy**2)
mask = rr > 2.5
image = jnp.exp(-0.5 * ((rr - 1.0) / 0.15)**2)
image = jnp.where(mask, 0.0, image)
noise = jnp.ones_like(image) * 0.02
psf = jnp.ones((5, 5)) / 25.0

sie = SIE(theta_E=1.0, e1=0.1, e2=0.0, center_x=0.0, center_y=0.0)
GRID_SIZES = [(20, 20), (30, 30), (40, 40), (50, 50)]

print("Benchmarking evidence evaluation time...")
print(f"Warmup: {N_WARMUP}, Repeat: {N_REPEAT}")
print(f"{'Grid':>10s} | {'Matrix (s)':>12s} | {'Operator (s)':>14s} | {'Ratio':>8s} | {'PCG iter':>10s}")
print("-" * 75)

results = []

for src_nx, src_ny in GRID_SIZES:
    pix_src = PixelizedSourceModel(nx=src_nx, ny=src_ny,
                                   regularization_type="first-order",
                                   log_lambda_reg=jnp.log(1.0))
    phys = PhysicalModel(lens_mass=[sie], source_light=[pix_src], lens_light=[])

    # --- Matrix backend ---
    pm_mat = PixelizedImageProbModel(image, noise, psf, dpix, phys, mask=mask)

    for _ in range(N_WARMUP):
        _ = pm_mat()

    times_mat = []
    for _ in range(N_REPEAT):
        t0 = time.perf_counter()
        _ = pm_mat()
        times_mat.append(time.perf_counter() - t0)
    t_mat = np.mean(times_mat)

    # --- Operator backend ---
    pm_op = PixelizedImageProbModelOperator(image, noise, psf, dpix, phys, mask=mask)

    for _ in range(N_WARMUP):
        _ = pm_op()

    times_op = []
    for _ in range(N_REPEAT):
        t0 = time.perf_counter()
        _ = pm_op()
        times_op.append(time.perf_counter() - t0)
    t_op = np.mean(times_op)

    ratio = t_op / t_mat if t_mat > 0 else float('inf')

    # Get PCG iteration count
    from TinyLensGpu.utils.cg_solver import pcg_solve
    # Run one more time to capture PCG info
    _, _, bx, by = pm_op.sim_obj._get_beta_sub_and_seed()
    xmi, xma, ymi, yma = pm_op.sim_obj._infer_and_fix_bbox(bx, by)
    from TinyLensGpu.utils.inversion.regularization import DenseRegularizationBuilder
    builder = DenseRegularizationBuilder(src_nx, src_ny, "first-order")
    xmif, xmaf, ymif, ymaf = float(xmi), float(xma), float(ymi), float(yma)
    reg_data = builder.make_reg_data(xmif, xmaf, ymif, ymaf)
    lam = jnp.asarray(1.0)
    block_chols, block_masks = pm_op.sim_obj.build_block_diag_preconditioner(
        pm_op.noise_1d, xmi, xma, ymi, yma, lam, builder, block_size=pm_op.block_size,
    )
    preconditioner = (block_chols, block_masks)
    A_data, _A_jit = pm_op.sim_obj.build_A_matvec(
        pm_op.noise_1d, xmi, xma, ymi, yma, lam, reg_data,
    )
    b = pm_op.sim_obj.build_rhs(
        pm_op.data_1d, pm_op.noise_1d, xmi, xma, ymi, yma,
    )
    _, info = pcg_solve(A_data, b, preconditioner, _A_jit, max_iter=200, rtol=1e-6)
    n_iter = int(info.n_iter)

    print(f"{src_nx}x{src_ny}  | {t_mat:12.4f} | {t_op:14.4f} | {ratio:7.2%} | {n_iter:10d}")

    results.append({
        'grid': f"{src_nx}x{src_ny}",
        'Ns': src_nx * src_ny,
        't_mat': t_mat,
        't_op': t_op,
        'ratio': ratio,
        'n_iter': n_iter,
    })

print()
print("Notes:")
print("  - Operator backend uses PCG (iterative) instead of Cholesky (direct).")
print("  - For small grids, explicit Cholesky is faster; PCG wins at larger Ns.")
print("  - The main benefit is memory, not CPU time, for moderate-sized problems.")
