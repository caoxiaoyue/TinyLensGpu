# Float32 Dense three-strength Nautilus posterior

User-requested follow-up on 2026-11-01: jointly sample the source strength
lambda_s and both potential strengths lambda4 and lambda0, then fit the
subhalo at the three marginal posterior medians. The algorithm remains
fixed-gradient single-step. The macro, 200×200 seed-17 subhalo observation,
noise, PSF, source mesh and potential mesh match the preceding scientific
experiments. Float64 observations and frozen truth gradients are cast to
Float32; geometry is constructed at the requested precision.

The inner likelihood analytically marginalizes signed source/potential
coefficients with complete Dense conditional Gaussian determinants. Cache
only strength-independent design, data Gram/RHS and prior templates. Source
precision is lambda_s times the existing second-order template; potential
precision is lambda4 D4ᵀD4 + lambda0 I. The three independent hyperpriors are
uniform in log10 coordinates: lambda_s [-8,-1], lambda4 [1,11], lambda0 [0,8].
Storage order is lambda_s, lambda4, lambda0; corner display order is lambda0,
lambda4, lambda_s.

The requested Float32 path uses `solve_rtol=1e-5`, highest JAX matrix-product
precision, full Cholesky solves and full determinants. When Float32 evidence
is nonfinite or fails its true residual certificate, the exact same candidate
is recomputed with the Float64 Dense formulation and `solve_rtol=1e-8`.
Thus valid hyperprior volume is not dropped on Float32 numerical failure.
This is explicitly a **Float32-primary, Float64-fallback** experiment. It does
not claim pure-Float32 validity over the broad prior or bitwise equality to
a uniformly Float64 numerical target.

Nautilus uses 300 live points, batches of 8, seed 271828, `f_live=0.01`, target
effective size 5000, and discarded exploration samples. A 33-point posterior
region comparison against the earlier independent Float64 run precedes
sampling. Every retained posterior point is re-evaluated in Float64 after
completion, with importance reweighting used to quantify precision effects.
Likelihood traces retain the original Float32 score, returned score, actual
residual and fallback flag. One progress-reporting interruption at exploration
completion was corrected; the persisted HDF5 sampler/RNG state is resumed
with unchanged numerical target and settings, as recorded in the manifest.

The conditional fields at marginal median strengths are distinct from a
maximum-evidence fit and from posterior-mean fields. They are reconstructed
through the public Dense fitter, then exported together with discrete truth,
raw source/potential fields, convergence, linear/nonlinear image predictions
and residuals. The recovery figure compares observed/linear image/noise
residual, true/recovered/source error, and discrete true/recovered/error κ.
Undefined support-only derivatives are masked. No clipping or affine
alignment is applied to reconstructed fields. Shared image and source
intensity scales and shared truth/recovered κ scales permit direct comparison.

## Reproduction

From the repository root:

```bash
source ~/anaconda3/bin/activate && conda activate tinylens_gpu
XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_ENABLE_COMPILATION_CACHE=false \
  python .scratch/potential-correction/experiments/sample_dense_three_strengths_precision.py \
  --output .scratch/potential-correction/output/three-strength-nautilus-float32-20261101/subhalo-nlive300 \
  --dtype float32 --n-live 300 --n-eff 5000 --batch-size 8
python .scratch/potential-correction/experiments/plot_dense_three_strength_median.py \
  .scratch/potential-correction/output/three-strength-nautilus-float32-20261101/subhalo-nlive300
```

Use a fresh output directory or `--resume` for a matching persisted sampler.
`--resume` now refuses to touch any file unless the persisted manifest's
scientific configuration, sampler settings, parameter order and recorded
precision policy all match the requested run; a missing run directory or
manifest fails without creating anything. Manifests written before the
explicit `precision_policy` field are still accepted: their policy is
reconstructed from the saved dtype and the historical literals
(`float32 -> solve_rtol 1e-5 with float64 fallback; float64 -> 1e-8,
matmul precision "highest"`), and a successful resumption records the
explicit policy alongside the resumption metadata. Manifests carrying an
explicitly different policy are rejected. `--median-reference-only`
reconstructs from the persisted manifest's scientific configuration rather
than the mutable `scientific_configuration.json`. The sampler driver
requires the earlier weighted Float64 posterior at
`research/three-strength-nautilus/posterior.npz` (relative to this feature)
for numerical validation only, not initialization or prior narrowing.

The figure script uses the Python Nature Figure alignment auditor. Pass
`--audit-scripts-dir DIRECTORY` to select the directory containing
`audit_panel_alignment.py`; without it the script searches
`~/.agents/skills/nature-figure/scripts` and
`~/.config/devin/skills/nature-figure/scripts`, and fails clearly when no
usable auditor is found. Add `--output DIRECTORY` to write summaries,
figures and audits into a NEW directory while reading the run inputs — the
safe way to re-render without overwriting frozen campaign products:

```bash
python .scratch/potential-correction/experiments/plot_dense_three_strength_median.py \
  .scratch/potential-correction/output/three-strength-nautilus-float32-20261101/subhalo-nlive300 \
  --output /path/to/new/figure-output
```

PNG (300 dpi), editable PDF and SVG are produced; journal-specific
submission formatting and TIFF export are outside this diagnostic delivery.

Note: the archived manifest, hashes and results in this directory describe
the historical run. Later driver changes only add reproduction safeguards
(state guards, read-only summarization, portable auditor lookup, separate
figure output) and fix the source-panel coordinate convention; they do not
recompute or alter the archived numerical evidence.

All checkpoints, likelihood traces, complete posterior, public-fit products,
figures and PDF audits are retained under the output directory. Compact
numerical summaries and the weighted posterior are retained beside this note.
The experiment leaves library code and public defaults unchanged.
