"""
Benchmark likelihood evaluation speed for the pixelized source model
with Matern-3/2 GP regularization.

Step 1: vectorized=False  — serial (single-sample) evaluation
Step 2: vectorized=True   — batched vmap evaluation with controlled batch size

Results are saved as JSON to output/.
"""

import os
import json
import time
from pathlib import Path

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

os.chdir(Path(__file__).parent)

import numpy as np
import jax
import jax.numpy as jnp

from TinyLensGpu.Inference import ParamU
from TinyLensGpu.PhysicalModel import PhysicalModel, EPL, Shear
from TinyLensGpu.PhysicalModel.LensImage.Pixelized.Light import PixelizedSourceModel
from TinyLensGpu.ObservationModel.LensImage.pixelized_image_model_operator import PixelizedImageProbModelOperator
from TinyLensGpu.Inference.build_prior import make_prior_transformation
from TinyLensGpu.Inference.build_likelihood import make_likelihood
from TinyLensGpu.utils import load_lens_data

# ------------------------------------------------------------------ #
# Configuration
# ------------------------------------------------------------------ #
DPIX        = 0.05
N_SERIAL    = 200   # number of single-sample calls for vectorized=False
N_BATCH     = 200   # total samples for vectorized=True
BATCH_SIZE  = 8     # batch size per vmap call (memory-safe)
N_WARMUP    = 3     # JIT warm-up calls

EPL_TRUE = dict(theta_E=1.0, gamma=2.2, e1=0.1, e2=0.0, center_x=0.0, center_y=0.0)
SHEAR_TRUE = dict(gamma1=0.05, gamma2=0.05)

# ------------------------------------------------------------------ #
# Data
# ------------------------------------------------------------------ #
print("[1] Loading data ...")
image_data, noise_map, psf_kernel, mask = load_lens_data(
    image_path="data/image.fits",
    noise_path="data/noise.fits",
    psf_path="data/psf.fits",
    mask_path="data/mask.fits",
)

# ------------------------------------------------------------------ #
# Physical model
# ------------------------------------------------------------------ #
print("[2] Building model ...")

epl = EPL(
    theta_E=ParamU("theta_E", EPL_TRUE["theta_E"],
                   prior_type="gaussian", prior_settings=[EPL_TRUE["theta_E"], 0.1],
                   limits=[0.3, 3.0]),
    gamma=ParamU("gamma", EPL_TRUE["gamma"],
                 prior_type="gaussian", prior_settings=[EPL_TRUE["gamma"], 0.1],
                 limits=[1.5, 3.0]),
    e1=ParamU("e1", EPL_TRUE["e1"],
              prior_type="gaussian", prior_settings=[EPL_TRUE["e1"], 0.1],
              limits=[-0.9, 0.9]),
    e2=ParamU("e2", EPL_TRUE["e2"],
              prior_type="gaussian", prior_settings=[EPL_TRUE["e2"], 0.1],
              limits=[-0.9, 0.9]),
    center_x=ParamU("center_x", EPL_TRUE["center_x"],
                    prior_type="gaussian", prior_settings=[EPL_TRUE["center_x"], 0.05],
                    limits=[-0.5, 0.5]),
    center_y=ParamU("center_y", EPL_TRUE["center_y"],
                    prior_type="gaussian", prior_settings=[EPL_TRUE["center_y"], 0.05],
                    limits=[-0.5, 0.5]),
)

shear = Shear(
    gamma1=ParamU("gamma1", SHEAR_TRUE["gamma1"],
                  prior_type="gaussian", prior_settings=[SHEAR_TRUE["gamma1"], 0.05],
                  limits=[-0.5, 0.5]),
    gamma2=ParamU("gamma2", SHEAR_TRUE["gamma2"],
                  prior_type="gaussian", prior_settings=[SHEAR_TRUE["gamma2"], 0.05],
                  limits=[-0.5, 0.5]),
)

pix_src = PixelizedSourceModel(
    nx=40,
    ny=40,
    regularization_type="matern32",
    lambda_reg=ParamU("lambda_reg", 1.0,
                      prior_type="log_uniform", prior_settings=[1e-3, 1e3],
                      limits=[1e-6, 1e6]),
    kernel_scale=ParamU("kernel_scale", 0.3,
                        prior_type="log_uniform", prior_settings=[0.01, 2.0],
                        limits=[1e-3, 10.0]),
)

phys_model = PhysicalModel(
    lens_mass=[epl, shear],
    source_light=[pix_src],
    lens_light=[],
)

epl.theta_E.to_dynamic()
epl.gamma.to_dynamic()
epl.e1.to_dynamic()
epl.e2.to_dynamic()
epl.center_x.to_dynamic()
epl.center_y.to_dynamic()
shear.gamma1.to_dynamic()
shear.gamma2.to_dynamic()
pix_src.lambda_reg.to_dynamic()
pix_src.kernel_scale.to_dynamic()

prob_model = PixelizedImageProbModelOperator(
    image_data=image_data,
    noise_map=noise_map,
    psf_kernel=psf_kernel,
    dpix=DPIX,
    phys_model=phys_model,
    mask=mask,
)

prior, prior_specs = make_prior_transformation(prob_model)
param_names = [s.name for s in prior_specs]
n_dim = len(param_names)
print(f"  {n_dim} dynamic parameters: {param_names}")

# ------------------------------------------------------------------ #
# Sample random unit-cube points and transform to physical space
# ------------------------------------------------------------------ #
rng = np.random.default_rng(42)
unit_samples = rng.uniform(0, 1, size=(max(N_SERIAL, N_BATCH), n_dim)).astype(np.float32)
phys_samples = np.array([prior(u) for u in unit_samples])

os.makedirs("output", exist_ok=True)

# ================================================================== #
# Step 1: vectorized=False
# ================================================================== #
print("\n" + "="*60)
print("Step 1: Benchmarking vectorized=False (serial evaluation)")
print("="*60)

loglike_serial = make_likelihood(prob_model, vectorized=False)

# Warm-up JIT
print(f"  Warming up JIT ({N_WARMUP} calls) ...")
for i in range(N_WARMUP):
    _ = loglike_serial(phys_samples[i])
jax.effects_barrier()

# Benchmark
print(f"  Running {N_SERIAL} serial calls ...")
t0 = time.perf_counter()
results_serial = []
for i in range(N_SERIAL):
    val = loglike_serial(phys_samples[i])
    results_serial.append(float(val))
jax.effects_barrier()
t1 = time.perf_counter()

elapsed_serial = t1 - t0
per_call_serial = elapsed_serial / N_SERIAL

print(f"  Total time : {elapsed_serial:.3f} s")
print(f"  Per call   : {per_call_serial*1e3:.2f} ms")
print(f"  Throughput : {N_SERIAL/elapsed_serial:.1f} calls/s")

bench_serial = {
    "mode": "vectorized=False",
    "regularization": "matern32",
    "n_calls": N_SERIAL,
    "total_time_s": round(elapsed_serial, 4),
    "per_call_ms": round(per_call_serial * 1e3, 4),
    "throughput_calls_per_s": round(N_SERIAL / elapsed_serial, 2),
    "loglike_mean": float(np.mean(results_serial)),
    "loglike_std": float(np.std(results_serial)),
    "device": str(jax.devices()[0]),
}

out_path_serial = "output/bench_matern32_serial.json"
with open(out_path_serial, "w") as f:
    json.dump(bench_serial, f, indent=2)
print(f"  Saved {out_path_serial}")

# ================================================================== #
# Step 2: vectorized=True
# ================================================================== #
print("\n" + "="*60)
print("Step 2: Benchmarking vectorized=True (batched vmap evaluation)")
print(f"  batch_size={BATCH_SIZE}, total samples={N_BATCH}")
print("="*60)

loglike_vec = make_likelihood(prob_model, vectorized=True)

n_batches = N_BATCH // BATCH_SIZE
assert n_batches > 0, "N_BATCH must be >= BATCH_SIZE"

# Warm-up JIT
print(f"  Warming up JIT ({N_WARMUP} batches) ...")
for i in range(N_WARMUP):
    batch = phys_samples[:BATCH_SIZE]
    _ = loglike_vec(batch)
jax.effects_barrier()

# Benchmark
print(f"  Running {n_batches} batches x {BATCH_SIZE} samples ...")
t0 = time.perf_counter()
results_vec = []
for i in range(n_batches):
    batch = phys_samples[i * BATCH_SIZE : (i + 1) * BATCH_SIZE]
    vals = loglike_vec(batch)
    results_vec.extend([float(v) for v in np.asarray(vals)])
jax.effects_barrier()
t1 = time.perf_counter()

elapsed_vec = t1 - t0
total_evals = n_batches * BATCH_SIZE
per_call_vec = elapsed_vec / total_evals
throughput_vec = total_evals / elapsed_vec

print(f"  Total time : {elapsed_vec:.3f} s")
print(f"  Per sample : {per_call_vec*1e3:.2f} ms")
print(f"  Throughput : {throughput_vec:.1f} samples/s")

speedup = per_call_serial / per_call_vec
print(f"  Speedup vs serial: {speedup:.1f}x")

bench_vec = {
    "mode": "vectorized=True",
    "regularization": "matern32",
    "batch_size": BATCH_SIZE,
    "n_batches": n_batches,
    "total_samples": total_evals,
    "total_time_s": round(elapsed_vec, 4),
    "per_sample_ms": round(per_call_vec * 1e3, 4),
    "throughput_samples_per_s": round(throughput_vec, 2),
    "speedup_vs_serial": round(speedup, 2),
    "loglike_mean": float(np.mean(results_vec)),
    "loglike_std": float(np.std(results_vec)),
    "device": str(jax.devices()[0]),
}

out_path_vec = "output/bench_matern32_vectorized.json"
with open(out_path_vec, "w") as f:
    json.dump(bench_vec, f, indent=2)
print(f"  Saved {out_path_vec}")

# ================================================================== #
# Summary
# ================================================================== #
print("\n" + "="*60)
print("Benchmark Summary")
print("="*60)
print(f"  {'Mode':<25s} {'Per-eval (ms)':>15s} {'Throughput (eval/s)':>20s}")
print(f"  {'-'*60}")
print(f"  {'serial (vectorized=False)':<25s} {per_call_serial*1e3:>15.2f} {N_SERIAL/elapsed_serial:>20.1f}")
print(f"  {'batched (vectorized=True)':<25s} {per_call_vec*1e3:>15.2f} {throughput_vec:>20.1f}")
print(f"  Speedup: {speedup:.1f}x")
print("="*60)
