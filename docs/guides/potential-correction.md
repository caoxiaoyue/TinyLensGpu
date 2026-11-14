# Pixelized source–potential reconstruction

Use potential correction to reconstruct a signed pixelized source and a local
lens-potential perturbation from one lens-light-subtracted image. The macro
mass model, source grid and bounding box, potential geometry, observation mask,
PSF and integration settings stay fixed during fitting and strength selection.
The supplied examples use mock truth for initialization. This release supports
reconstruction and algorithm validation; it does not provide calibrated
subhalo detection significance, mass intervals or posterior-sampling results.

## Choose an algorithm and backend

| Entry point | Initialization | Fitted image |
| --- | --- | --- |
| `SingleStepPotentialCorrection` | A source-gradient callable evaluated once at macro image-subpixel rays, then frozen; scientific mocks use analytic truth gradients | A linear source–potential prediction, plus a separate nonlinear re-render |
| `IterativePotentialCorrection` | Truth brightness sampled at the fixed **source nodes**, with zero initial potential | A nonlinear image; mappings and gradients of the selected interpolant refresh after accepted updates |
| `LBFGSPotentialCorrection` | Supplied signed intensities on the fixed source nodes, with zero initial potential | A cubic Hermite image optimized with L-BFGS-B at fixed strengths |

A documented small-step derivative of the truth can also supply the single-step
gradient callable. All three fit the complete signed source and potential
together. Source brightness is not fixed to truth, and negative coefficients
are neither clipped nor
constrained by NNLS/PNPG. The existing source–lens-light “joint semi-linear
inversion” is a separate workflow with its established block-Schur contract.
No lens-light amplitudes or macro parameters are fitted here.

Use `backend="dense"` for complete determinants and
`backend="operator"` for fully coupled matrix-free PCG. Bounded principal
blocks approximate its preconditioner and determinants, while the actual solve
retains all source–potential cross terms. A partition suitable for one geometry
or initialization is not automatically calibrated for another.

## Minimal Python use

This small representable nodal model is a runnable numerical example, not a
scientific prior or recovery calibration. It uses explicit float64 inputs.
Create the simulator, coordinates, arrays and bounds inside the scoped x64
context: promoting an already rounded float32 input cannot recover its original
precision. The renderer's dtype selection does not undo earlier rounding.

```python
import jax
import jax.numpy as jnp
from TinyLensGpu.ForwardSimulation import PotentialCorrectionRenderer, SimulatorConfig
from TinyLensGpu.PhysicalModel import PhysicalModel, Shear
from TinyLensGpu.ObservationModel import (
    SingleStepPotentialCorrection, IterativePotentialCorrection,
)

with jax.experimental.enable_x64(True):
    macro = PhysicalModel(
        lens_mass=[Shear(gamma1=0.0, gamma2=0.0)],
        source_light=[], lens_light=[],
    )
    config = SimulatorConfig(0.1, 4, jnp.ones((1, 1)), nsub=1)
    renderer = PotentialCorrectionRenderer(
        macro, config, source_n=2,
        source_bbox=(-0.5, 0.5, -0.5, 0.5), dtype="float64",
    )
    sx, sy = jnp.meshgrid(renderer.source_x_axis, renderer.source_y_axis)
    initial_source = (0.775 + 0.95*sx + 0.05*sy + 0.5*sx*sy).ravel()
    grid = renderer.potential_grid
    truth_potential = 0.003*jnp.exp(-((grid.x-0.07)**2 + (grid.y+0.03)**2)/0.02)
    image = renderer.render(initial_source, truth_potential).model_image
    image = image + 0.002*jnp.sin(jnp.arange(16)).reshape(4, 4)
    noise = jnp.full_like(image, 0.07)

    def truth_gradient(beta_x, beta_y):
        return 0.95 + 0.5*beta_y, 0.05 + 0.5*beta_x

    single = SingleStepPotentialCorrection(
        renderer, image, noise, truth_gradient,
        lambda_amplitude=20.0, source_regularization="zero-order",
    )
    iterative = IterativePotentialCorrection(
        renderer, image, noise, initial_source,
        lambda_amplitude=20.0, source_regularization="zero-order",
        initialization_provenance={"source": "analytic truth at fixed source nodes"},
    )
    single_result = single.fit(lambda_source=0.3, lambda_potential=0.02)
    result = iterative.fit(lambda_source=0.3, lambda_potential=0.02)
    assert single_result.success and result.success
```

The physical strengths passed to `fit` are positive **direct values**.
`lambda_potential` multiplies fourth differences. The single-step and GN/LM
constructors fix `lambda_amplitude` for their fits and search candidates.
`source_regularization` defaults to `"second-order"`; existing nonadaptive
`"zero-order"` and `"first-order"` are also supported. Full native image and
noise arrays must match the simulator shape. Only selected pixels must be
finite, and selected noise standard deviations must be positive. Masks use
`True` for excluded pixels.

Select `source_interpolation="cubic_hermite"` on the renderer to extend
source support by two grid spacings per side using fixed zero exterior nodes.
The default `"bilinear"` uses the fitted box as its support. Single-step
Hermite applies its frozen callback strictly inside the extended support and
sets it to zero at the outer edge and beyond.

For a reconstructed macro source, freeze the gradient of the selected interpolant with
`renderer.source_spatial_gradient(source_values, renderer.source_mapping())`
and supply a callable returning that pair. The single-step constructor also
accepts `source_scale`, a fixed positive vector of length `source_n**2`.
This reuses the existing edge-weighted adaptive finite differences in both
the penalty and the normalized prior determinant; only `lambda_source`
changes during strength selection. Preserve the macro fit's template and
adaptive parameters when exporting that scale, rather than recalculating it
from each new jointly reconstructed source. The [SLACS J0946 workspace](../../workspace/slacs0946_potential_correction/)
illustrates the M3 handoff, a single-step three-strength Nautilus search with
the dense Float32 backend, and an iterative L-BFGS-B fit.

For the operator backend, add `backend="operator"`, `block_size`, optional
`block_indices`, `pcg_max_iter`, `pcg_rtol` and `pcg_atol`. An explicit partition
covers every joint coefficient exactly once; blocks can mix fields but cannot
contain the whole potential or joint field. Default bounded blocks separate
fields. The result records complete indices, sizes and a partition digest.
The scientific examples check partition provenance before automatic reuse,
including traced macro rays and initial gradients for source-incidence votes.

Every iterative call resets source nodes, potential and damping. Its controls
include `max_iter`, initial/minimum/maximum damping and increase/decrease
factors, `max_rejections`, objective/update/gradient tolerances and stall limits.
Actual nonlinear objective decrease governs acceptance. The prior acts on the
accumulated fields; iteration is not repeated single-step fitting followed by
adding its full solution. A rejected proposal cannot replace accepted fields.
A small damping-dominated step does not establish convergence, and unresolved
bilinear kinks can return an explicit stall.

## Fixed-strength L-BFGS-B

Use the preceding example's observation, macro and initial source with a
renderer configured for `cubic_hermite`. Both backends use the renderer's
`float32` or `float64` dtype for JAX computations, including the objective,
whitening and evidence. SciPy keeps its optimizer state in Float64. Supply all
three physical strengths with each fit.

```python
from TinyLensGpu.ObservationModel import (
    LBFGSPotentialCorrection, PotentialCorrectionStrengths,
)

hermite = PotentialCorrectionRenderer(
    macro, config, source_n=2, source_bbox=(-0.5, 0.5, -0.5, 0.5),
    dtype="float64", source_interpolation="cubic_hermite",
)
fitter = LBFGSPotentialCorrection(
    hermite, image, noise, initial_source, source_regularization="zero-order",
)
strengths = PotentialCorrectionStrengths(0.3, 0.02, 20.0)
result = fitter.fit(strengths)
```

Each call resets the supplied source and zero potential. The optional
`source_scale` is a fixed finite positive weight for each fitted source node.
The default whitening reference is the starting Hermite source at zero
potential. Pass `whitening_reference=PotentialCorrectionWhiteningReference(values, "bilinear")`
to use a distinct reference on the same grid with hard-box bilinear
interpolation. Both the fit and endpoint evaluation use the Hermite image.

The dense backend uses data-space whitening and the complete local GN
determinant ratio. To use bounded principal blocks for whitening and
determinants, call `fitter.fit(strengths, backend="operator")`. The operator
defaults to field-separated contiguous blocks of at most 64 coefficients and
accepts the partition and PCG controls above.

The default `gn_stationarity_tol=10000` effectively disables rejection by
GN magnitude for typical fits. In a matched J0946 Float32 operator search,
the relaxed setting qualified six candidates above the former `0.01` cutoff
and recovered the upper-arc convergence peak at the same grid node. Its
largest qualified GN value was `0.04445`, well below the finite default cap.
Set `gn_stationarity_tol=0.01` to use the previous stricter cutoff.
[J0946 threshold comparison](../../.scratch/lbfgs-potential-correction/research/j0946-gn-threshold.md)

Both backends require native optimizer success, finite endpoints, and a GN
stationarity statistic `0.5 * g.T @ H_GN^{-1} @ g` at or below the configured
cutoff. Here `g` is the physical objective gradient and `H_GN` is the undamped
coupled Gauss–Newton curvature at the endpoint. The operator uses PCG and
checks its recomputed residual against the constructor's `solve_atol` and
`solve_rtol`. Read the native status, evaluation counts and stationarity in
`optimizer_info`. The operator certificate is in
`optimizer_info.diagnostics["stationarity_solver"]`. L-BFGS-B sets
`solver_info=None`. The optimizer iteration limit defaults to `max_iter=3000`.

## Geometry, derivatives and the physical prior

The source is a fixed square grid in row-major y-then-x ordering. Existing
bilinear cell selection, endpoint branches and out-of-bounds zeros are
preserved; the source box never expands automatically. Source gradients
differentiate the same brightness interpolation within its selected cell.

Core potential nodes **C** are the four coarse-grid interpolation corners
needed by the subpixels in the PSF render region. This includes fitted pixels
and any other pixels in the cutout that can contribute to them through the PSF.
Fitted unknowns live on
**U**, the union of C and its axial one- and two-node neighbors. Nodes in U
outside C are fitted support unknowns, including those outside the observed
arc; they are not fixed-zero padding or additional observations. The observation
mask is preserved, including holes, disconnected arcs and isolated pixels.
Incomplete explicit geometry raises a validation error. When migrating an
explicit grid, construct it from the new renderer's `query_x` and `query_y`
using the desired spacing, origin, and any extra support nodes.

Potential spacing is independent of source resolution and defaults to twice
the native image-pixel spacing. Centered physical derivatives use angular
spacing. Core deflection is bilinearly interpolated to subpixels and subtracted
from the macro source coordinates. The same map is used in linear responses,
current Jacobians and nonlinear rendering. The convergence correction is
`delta_kappa = (delta_psi_xx + delta_psi_yy)/2`. Derivative maps are valid only
on C: `derivative_valid` marks that domain and support-only entries are NaN.
Subpixel averaging and zero-extended PSF convolution use the same image
operator as ordinary pixelized modeling. At zero potential correction the
predictions agree on the fitted pixels for the same macro, source and sampling.
Full-image diagnostics show the convolution of the modeled render region;
only the fitted pixels enter the likelihood.

Smoothing enumerates each complete horizontal/vertical five-node window lying
wholly in U, once, with pixel-unit coefficients `[1,-4,6,-4,1]`. It neither
bridges absent nodes nor falls back to lower-order differences at boundaries.
The complete potential precision is

\[
R_\psi=\lambda_4(D_{4x}^{\mathsf T}D_{4x}+D_{4y}^{\mathsf T}D_{4y})+\lambda_0I.
\]

Here `lambda_potential=lambda4` and `lambda_amplitude=lambda0`. The positive
amplitude prior applies to **every** potential unknown and participates in
normalization. It is not numerical jitter or a pure-strength determinant
shortcut. It stays independent of lambda4 and fixed during smoothing searches.
No hard constant/slope gauge constraints are imposed. `potential_window_coverage`
records how many fourth windows involve each node. A proper prior does not
establish that every support coefficient or broad potential mode is measured.

## Evidence and result interpretation

| Algorithm/backend | `evidence_method` | `determinant_method` | `evidence_is_exact` |
| --- | --- | --- | --- |
| Single-step dense | `conditional_linear_gaussian` | `full_cholesky` | True, conditional on the fixed linearization and declared prior |
| Single-step operator | `conditional_linear_gaussian` | `principal_block_cholesky` | False |
| Iterative dense | `local_gauss_newton` | `full_cholesky` | False |
| Iterative operator | `local_gauss_newton` | `principal_block_cholesky` | False |
| L-BFGS-B dense | `local_gauss_newton` | `full_data_space` | False |
| L-BFGS-B operator | `local_gauss_newton` | `principal_block_cholesky` | False |

The single-step Gaussian formula uses its fitted **linear** residual. Its
nonlinear re-render measures linearization error and does not replace that
residual in evidence. GN/LM and L-BFGS-B evidence use the converged nonlinear
residual and a freshly evaluated, **undamped** Gauss–Newton curvature. Local
linearization and block determinants are separate approximations. Agreement
of fitted fields does not make absolute evidences or selected strengths
interchangeable. The [small nested validation](../validation/potential-correction-iterative-inference.md)
retains both an accepted partition and a rejected one with misleading recovery.

`PotentialCorrectionResult` carries raw source/potential fields and geometry,
valid deflection/convergence maps, selected data/noise, strengths, chi2,
regularization penalty, objective, evidence terms/methods, dtype, status,
configuration and initialization provenance. Single-step results contain
`linear_*` and `nonlinear_*` image/residual products. GN/LM and L-BFGS-B
`linear_*` products are **None** and the nonlinear image defines chi2/objective.
Full residual maps are NaN at excluded observation pixels.

Check `success` and `termination_reason` before using evidence. Invalid
configuration raises an exception; valid numerical failure returns a failed
result with `log_evidence=None` and the last available accepted fields.
Iteration/rejection/damping limits, stalling, nonfinite states and unsuccessful
inner solves are failures. `solver_info` describes the actual last linear
solve, separately from nonlinear success: it can be converged while the fit
stalls. History includes initial objective and accepted/rejected trial records,
`objective_before/trial/after`, damping, predicted/actual decrease, scaled
updates/gradients and solver diagnostics. Final refreshed gradient metadata
is separate from a trial's pre-step gradient.

Recursive PCG stopping (`pcg_rtol/pcg_atol`) is separate from recomputed true
residual acceptance (`solve_rtol`, and iterative constructor `solve_atol`).
The iterative certificate is `norm(A*delta-rhs) <= solve_atol +
solve_rtol*norm(rhs)`. After recursive convergence, the iterative operator may
make **one** separate residual correction using the remaining total PCG budget,
the same current damped operator and bounded factors. Only a finite improving
candidate can be selected. `solver_info.iterations` aggregates all phases;
`solver_info.diagnostics` records phase counts, recursive/true residuals,
refinement decisions, original RHS norm and the physical threshold. The legacy
recursive scalar describes the initial phase on the original RHS. Budget
exhaustion or an unsatisfied true certificate remains failure. Refinement does
not alter the prior, LM acceptance or final undamped evidence.

## Strength selection and reproducible examples

For a single-step or GN/LM fitter, the inference API can select source and
potential strengths.

```python
from TinyLensGpu.Inference import (
    PotentialCorrectionEvidence, optimize_potential_correction,
)

# Reuse `iterative` from the minimal example above.
search = optimize_potential_correction(
    iterative, backend="dense",
    log_bounds=((-2.302585092994046, 0.6931471805599453),
                (-6.907755278982137, 0.0)),  # natural logs: source [.1,2], D4 [.001,1]
    fit_options={"max_iter": 240}, maxfun=100, len_tol=1e-3,
)
if search.has_valid_fit:
    best = search.reconstruction  # retained public result; no additional fit
```

`PotentialCorrectionEvidence(problem, backend=..., fit_options=...)` is a
callable negative evidence over `[log_lambda_source, log_lambda_potential]`.
Every candidate is a complete fit with the same initialization/geometry and
fixed amplitude prior. Failed candidates return positive infinity and retain
termination/solver reasons; configuration errors propagate. An all-invalid
search has no reconstruction. `search.has_valid_fit` is distinct from
`optimizer_success` and combined `success`: DIRECT's approximate evaluation
budget can leave a useful converged fit while optimization is incomplete.
Boundary sides report fractional distance to an edge of the explicit natural-log
range. A sampled boundary optimum is a reported limitation, not an interior
optimum or a global-optimality certificate.

Activate `tinylens_gpu`, use the repository root, set preallocation off, and
choose fresh output directories to preserve existing evidence:

```bash
source ~/anaconda3/bin/activate && conda activate tinylens_gpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export MPLBACKEND=Agg
python examples/potential_correction/iterate_fixed_strength.py --backend operator
python examples/potential_correction/tune_iterative.py --backend operator \
  --output output/pc_nested_new
python examples/potential_correction/single_step_scientific.py \
  --config examples/potential_correction/scientific_configuration.json \
  --backend both --output output/pc_single_new
python examples/potential_correction/iterative_scientific.py \
  --config examples/potential_correction/iterative_scientific_configuration.json \
  --backend both --mode fixed --single-step-reference output/pc_single_new \
  --output output/pc_iterative_new
python -m examples.potential_correction.lbfgs_mock \
  --backend both --output output/pc_lbfgs_mock_new
```

Choose a new output directory for each L-BFGS-B mock run. The first two
commands are small numerical demonstrations. Scientific runners also
support `--mode tune`, but the retained 200×200 results are **fixed-strength**,
not claims of converged scientific outer searches. Mock generation and
single-step tuning commands are in the [example README](../../examples/potential_correction/README.md).
No PyAuto runtime or reference environment is needed.

Scientific exports contain safe `reconstruction.npz`, `result.json`, complete
configuration/provenance/history and diagnostic figures. The shared
`examples.potential_correction.result_products.export_result(...)` consumes a
public result plus explicit observation/truth context. Failure exports retain
`failure.json` and histories and remove the exporter's known stale success
products. Restore a successful public result without fitting:

```python
from examples.potential_correction.result_products import load_result
saved = load_result("output/pc_iterative_new/subhalo/dense")
assert saved.success and saved.linear_model_image is None
```

Loading a failure as a success raises an error. Replotting/new truth diagnostics
still require explicit observation/truth context. The compact nested numerical
example uses its own `nested_result.json/npz` format. Raw fields always remain
available. Optional affine-aligned truth comparisons are named diagnostics;
they do not replace fields, images, priors or evidence. Analytic point-sampled
truth and derivatives of sampled nodal truth are separately identified, including
the Pseudo-Jaffe central singularity convention.

## Delivered science, precision and resource limits

The scientific configuration is 200×200 at 0.012 arcsec/pixel, nsub=2 for both
mock and inversion, a 27×27 PSF with sigma 0.05 arcsec, noise 0.01/seed 17,
source49 on `[-0.4,0.4]^2`, potential spacing 0.075, 22,016 data, 724 core
and 956 potential unknowns. It uses `PseudoJaffeSubhalo(b_sub=0.04)` at
`(0.75,0.1)`, with host theta_E=0.8 and a matched no-subhalo control. This is
separate from the generator default b_sub=0.01 and from the 64×64 calibration.
Fixed lambda_s is 0.0006, lambda4 is 1.6e7/1e11 for subhalo/control, and
lambda0 is 10000. The configuration records example-only iterative controls:
subhalo/control gradient atol 5e-4/3e-5 and true residual
`5e-5 + 1e-8*norm(rhs)`. Public defaults are unchanged.

Float64 is recommended for this delivered stiff scientific model. Localized
kappa recovery is useful, with substantial peak suppression and imperfect broad
modes. The measured lambda0 **10000→30000** comparison has stable local kappa
but sensitive raw potential/deflection and changing evidence. Lambda0=3000
failed with null evidence; lower-side convergence remains unverified. These
results do not establish a universally weak amplitude prior. Read the
[iterative science report](../validation/potential-correction-iterative-scientific.md)
and [single-step report](../validation/potential-correction-single-step-scientific.md)
for actual configurations, raw/affine errors and failed probes.

A later [peak-bias study](../validation/potential-correction-peak-recovery.md)
uses a shared finite evidence-selection rule and stricter common controls.
For the reference mock it raises nonlinear peak recovery from 48.08% to
85.30% while preserving the original small control. The subhalo background
variation and whole-core error increase, so this is an optional study workflow
with a measured trade-off; it does not replace the fixed-strength defaults.
The report includes its declared noise-seed validation policy and reproduction
commands.

Float32 is validated for the named **coupled37 / small32** profile: source2,
4×4 observations, 33 potential nodes, zero-order source prior and strengths
(0.3,0.02,20). Explicit controls are solve_rtol=1e-5, iterative solve_atol=1e-6,
PCG rtol=1e-8/budget400, gradient atol/rtol=1e-4/1e-5, objective_rtol=1e-6
and step_rtol=1e-4. Requested dtype applies to caller inputs, renderer and
subsequent arithmetic inside the appropriate scoped x64 context; surrounding
process defaults are preserved. The unchanged field guard is absolute-plus-
relative atol=rtol=2e-4 against strict same-method float64, with a separate
0.002 evidence budget. This profile is not transferred to scientific grids.
All eight matched scientific float32 fits failed their numerical gates and
have null evidence; their elapsed times are failure diagnostics.

The [precision/performance report](../validation/potential-correction-precision-performance.md)
provides reproducible phase commands, supported profile invocation, rejected
calibration, 32/64/200 image and source-size axes, synchronized first/warm/cold
workflow costs, and allocator versus sampled process memory. First-fit time
includes compilation and lazy preparation; first-minus-warm is not pure compiler
time. The scientific layout shows no universal operator speed or memory
advantage, and float32 is not uniformly faster even in the supported small case.
Do not extrapolate the synthetic allocation probe or parametric-model timings
to these workflows.

For implementation/testing provenance and remaining review gates, use the
[requirement coverage and integration record](../validation/potential-correction-coverage.md).
