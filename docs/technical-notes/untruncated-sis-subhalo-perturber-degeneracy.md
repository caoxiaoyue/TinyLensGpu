# Untruncated SIS subhalo perturbers: a degeneracy with the main lens

## Scope and conclusion

This note records what goes wrong when a perturbing mass component (a "subhalo"
in a gravitational-imaging search) is modeled with TinyLensGpu's
[`SIS`](../../TinyLensGpu/PhysicalModel/LensImage/Parametric/Mass/sis.py), and
what to model instead.

**The profile is wrong for this role.** `SIS` returns a deflection of constant
magnitude `theta_E` at *every* radius:

```python
# TinyLensGpu/PhysicalModel/LensImage/Parametric/Mass/sis.py
scale = theta_E / radius
return x_local * scale, y_local * scale        # |alpha| = theta_E, always
```

A real subhalo is tidally truncated, so its influence is local. An untruncated
SIS has no such scale: a perturber placed 4.5 arcsec from the lens still
deflects the arc by its full `theta_E`. Over the ring the resulting deflection
field is **85-99.9% expressible as a constant vector plus a linear gradient** —
the same first-order field the main lens's own `theta_E`, centre and external
shear already generate. "Add a distant/bigger SIS and move the main lens" is
therefore almost free, and two things follow:

1. A smooth-vs-perturbed Bayes factor computed by widening the shared smooth
   prior **never converges**: the perturbed arm keeps buying likelihood with the
   extra volume (measured: `delta ln Z` = +26.0, +48.2, +81.9 as the shared prior
   went 5, 10, 20 `sigma`).
2. The recovered perturber **mass is not identifiable** and its **position runs
   away** from the arc: with a full-image position prior the posterior put
   *zero* weight inside the inner ring where the known perturber sits, and all
   of its weight at 2.1-4.5 arcsec, while the recovered `theta_E_sub` grew from
   0.015 to 0.041 as the prior widened.

**What to model instead:** a truncated profile (pseudo-Jaffe / PIEMD, truncated
NFW, or any profile whose deflection decays beyond a truncation radius), with a
truncation radius that is either sampled or tied to the perturber's scale.
Localising the position prior is a *diagnostic* that reveals the known mode; it
is not a fix, because the degeneracy lives on the mass side.

The measurements below come from the SLACS J0946+1006 smooth-vs-subhalo
comparison in
[`examples/pix_src_demo_operator/pipe/subhalo_0946/model.py`](../../examples/pix_src_demo_operator/pipe/subhalo_0946/model.py),
whose full record lives in the untracked scratch directory
`.scratch/subhalo-prior-audit/` (`sensitivity-study.md` plus the probes listed
in §6).

## 1. Why "constant deflection" is the problem

For a singular isothermal sphere the deflection angle is
`alpha(theta) = theta_E` for all `theta`; this is not an artifact of the
implementation, it is the property that makes the SIS analytically convenient
(and it is fine for a *main* lens, whose mass genuinely dominates and extends
far beyond the ring). Two comparisons make the modeling consequence explicit:

| Profile | Deflection at radius `b` from the perturber | Effect on an arc 4 arcsec away |
| --- | --- | --- |
| Point mass | `theta_E^2 / b` | negligible |
| Truncated (pseudo-Jaffe, `r_t` = 2-3 `b`) | `~ theta_E` for `b < r_t`, decaying beyond | local |
| TinyLensGpu `SIS` | `theta_E` for every `b` | **full `theta_E`** |

The failure mode is therefore not "the perturbation is too small to detect" but
the opposite: the perturbation does not decay, so it can be traded against
parameters that describe the *global* mass distribution.

## 2. Quantifying the degeneracy

Project the perturber's deflection field over the arc onto
`span{1, x, y}` per component (3 basis functions per component, 6 parameters
total) and report the explained fraction of the field's power. That span is a
first-order proxy for the field the main lens's own parameters can generate over
the ring: a translation of the lens centre, a change in `theta_E` (radially
uniform to first order across a narrow annulus), external shear (exactly linear
in `x, y`), and ellipticity.

```python
def constant_plus_gradient_fraction(points, field):
    design = np.column_stack([np.ones(len(points)), points[:, 0], points[:, 1]])
    residual = 0.0
    for comp in range(2):
        coeff, *_ = np.linalg.lstsq(design, field[:, comp], rcond=None)
        residual += np.sum((field[:, comp] - design @ coeff) ** 2)
    return 1.0 - residual / np.sum(field ** 2)
```

For a perturber at position `p`, the field is `theta_E_sub * (q - p)/|q - p|`
sampled over the arc annulus. On J0946's inner-ring annulus
(1.2-1.6 arcsec, `R_E1 = 1.43` arcsec):

| Perturber radius \|p\| (arcsec) | Explained by `{constant + gradient}` | Residual (genuine local signature) |
| ---: | ---: | ---: |
| 0.50 | 96.2% | 3.8% |
| 0.80 | 91.8% | 8.2% |
| 1.00 | 88.3% | 11.7% |
| **1.32** (on the ring) | **84.6%** | **15.4%** |
| 1.60 | 90.7% | 9.3% |
| 2.00 | 96.2% | 3.8% |
| 2.50 | 98.4% | 1.6% |
| 3.00 | 99.2% | 0.8% |
| 4.00 | 99.8% | 0.2% |
| 4.50 | 99.85% | 0.15% |
| 6.00 | 99.95% | 0.05% |

The degeneracy is *strongest at the ring* and *complete outside it*, and — the
crucial point — it never becomes expensive, because the amplitude does not decay
with radius either. There is no radius at which "a distant SIS" stops being an
almost-free re-parameterisation of the main lens.

A useful corollary for interpreting the recovered mass: the fraction above is
also the fraction of the perturbation that a perturber's *mass* can supply
without any local signature. Near the ring 15% of the field is irreducible; at
2 arcsec and beyond, essentially nothing is.

## 3. What it does to a real smooth-vs-perturbed comparison

Historical SIS-run configuration: 201 x 201 pixels at `dpix = 0.05` arcsec, circular modeling mask
of radius 2.5 arcsec, pixelized source, `log_lambda_reg` log-uniform of width
1.0 dex centred on the regularisation-calibration stage, `adaptive_reg_rho`
fixed, and both compared arms reading one frozen specification
(`comparison_prior.json`) so that every common parameter has a byte-identical
prior. The null arm is the smooth EPL + shear model; the perturbed arm adds one
`SIS` perturber.

### 3.1 The evidence does not converge

Widening the shared smooth prior by a factor 2 per mass parameter (8 parameters,
so `ln(V_wide/V_narrow) = 8 ln 2 = 5.5452`) leaves the null arm falling by
exactly that amount — the expected response for a sharply peaked likelihood, and
the sanity check that both runs share one integrand. The perturbed arm instead
*gains* relative to the nesting bound
`ln Z(wide) >= ln Z(narrow) - n_freed ln f`:

| Position prior | Shared floor | `ln Z` null | `ln Z` perturbed | `delta ln Z` | Perturbed arm gain over the bound |
| --- | --- | ---: | ---: | ---: | ---: |
| full image | 5 `sigma_M3` | 5778.463 | 5817.380 | +38.92 | — |
| full image | 10 | 5772.920 | 5836.652 | +63.73 | +24.8 |
| full image | 20 | 5767.392 | 5860.717 | +93.33 | +29.6 |
| localised (0.2 arcsec) | 5 | 5778.467 | 5804.514 | +26.05 | — |
| localised (0.2 arcsec) | 10 | 5772.930 | 5821.093 | +48.16 | +22.1 |
| localised (0.2 arcsec) | 20 | 5767.384 | 5849.326 | +81.94 | +33.8 |

(The gain column is per doubling of the shared volume, i.e. relative to the
nesting bound evaluated at the next-narrower run. A quantile+padding prior at
~2.9 `sigma_M3` — the narrowest configuration tried — gave +26.5 for the
full-image series, so the series starts well above zero and grows from there.)

### 3.2 The perturbed arm's smooth solution drifts, and tracks the prior edge

Because the far-field perturber can substitute for the main-lens parameters, the
perturbed arm's preferred smooth solution is not the smooth arm's solution, and
the shared prior decides how far it may move:

| Configuration | `theta_E` drift from the smooth solution | Posterior weight on that edge |
| --- | ---: | ---: |
| full image, 5 `sigma_M3` | -5.8 `sigma_M3` | 57.6% |
| full image, 10 | -9.9 | 79.0% |
| full image, 20 | -19.7 | 76.2% |
| localised, 5 | -4.85 | 46.9% |
| localised, 10 | -9.85 | 72.7% |
| localised, 20 | -19.80 | 88.3% |

The perturbed arm's `theta_E` posterior lower quantile sits *on* the prior's
lower edge in every run (e.g. localised 20 `sigma`: prior `[1.3831, 1.4175]`,
q16/q50/q84 = `[1.3832, 1.3835, 1.3840]`). This **edge-tracking** behaviour is
the cheapest convergence test available: if the perturbed arm stays the same
fraction of its own posterior sigma from the edge while the drift grows in
proportion to the width, the prior is clipping it and there is no width at which
widening stops helping.

### 3.3 The compensation is a pair, not a single parameter shift

Evaluating the real likelihood with the subhalo switched off separates the two
halves of the compensation (values are `delta ln L` against the perturbed arm's
own median):

| Quantity | Value |
| --- | ---: |
| Perturber's contribution at fixed smooth parameters | **+121.0** |
| Cost of the smooth-parameter shift, on the null model | **-81.8** |
| Net likelihood gain at the joint optimum | +39.2 |
| Perturbed arm at `theta_E_subhalo = 0` vs the null arm | identical (invariance check) |

So the smooth parameters do not drift because they fit better — on their own
they fit *worse*. They drift because the perturber **pays for** the drift, and
the pair together reaches a likelihood neither reaches alone. That is the
signature of a degeneracy being exploited, not of a detected perturber: the
"mass" being reported is partly a main-lens parameter in disguise. In the wide
runs the drift is not confined to `theta_E` either — at 20 `sigma_M3` the
perturbed arm moved `theta_E` by -19.8 `sigma_M3`, the lens centre by +18.1, the
external shear by +3.3 and the ellipticity by +5.7 simultaneously.

### 3.4 A full-image position prior does not contain the known perturber

For the J0946 subhalo, detected in previous work and sitting on the inner ring,
the full-image position posterior gave (three independent runs):

| Run | Radial median of the position posterior | Weight inside `r < 1.43` arcsec | Weight at `r > 1.9` arcsec | `theta_E_sub` |
| --- | ---: | ---: | ---: | ---: |
| full image, 5 `sigma_M3` | 2.13 arcsec (1.69, 4.43) | **0.000** | 0.06 | 0.0152 |
| full image, 10 | 2.27 arcsec (1.87, 4.48) | **0.000** | 0.57 | 0.0235 |
| full image, 20 | 2.53 arcsec (2.18, 4.55) | **0.000** | 1.00 | 0.0414 |

The known-position mode is not merely subdominant, it is *absent* from the
posterior, and part of the weight sits outside the modeling mask (2.5 arcsec).
The recovered "detection" is a perturber parked in the outer region with a mass
that grows with the allowed prior volume.

### 3.5 Localising the position: what it fixes and what it does not

Re-running with a truncated-Gaussian position prior (sigma = 0.2 arcsec,
truncated at 3 sigma per axis, effective search area `2 pi sigma^2 = 0.251`
arcsec^2 instead of the full 101.0 arcsec^2) recovers the known perturber:

* position (-0.83, +1.03), (-0.77, +1.04), (-0.74, +1.04) arcsec across the
  three floors — all within 0.6 sigma of the reference residual, and interior to
  the prior rather than at its edge;
* `theta_E_sub = 0.0122 (0.0110, 0.0134)`, with `P(theta_E_sub > 0.02) = 0.000`
  at the narrowest floor;
* 88% of the position posterior has `r < theta_E = 1.396` arcsec, i.e. a genuine
  sub-critical-curve perturber rather than an external perturber.

But it does **not** fix the evidence: `delta ln Z` still doubles twice
(+26.0 -> +48.2 -> +81.9), because with the position pinned the perturber simply
answers a wider smooth prior by *growing* instead of moving
(`theta_E_sub` 0.0122 -> 0.0196 -> 0.0330). Only the position is robust; the mass
and the evidence remain width-dependent, i.e. not yet measurements.

A related check worth running before quoting any per-cell result: whether the
recovered position sits on a ring with real emission. Compute the radial SNR
profile directly from the FITS image. On J0946 the inner-ring annulus
(1.2-1.8 arcsec) has mean SNR 10.9-18 and peak 30-35, while the 2.0-2.4 arcsec
annulus has mean 3.7 and peak 9.6 — statistically indistinguishable from the
control annulus *outside* the modeling mask (mean 3.8, peak 7.7). The outer ring
is real but faint, and this noise map is optimistic there, so a perturber landing
in that region needs false-positive calibration on mocks before it means
anything.

## 4. Recommended practice

1. **Model the perturber with a truncated profile.** Use a pseudo-Jaffe/PIEMD or
   truncated NFW, with a truncation radius that is either sampled or tied to the
   perturber's scale (`r_t ~ 2-3 b` is a common choice). The requirement is that
   the deflection magnitude decays beyond `r_t`; the shape requirement
   (non-degeneracy with `{constant, gradient}`) is then automatic, because a
   perturber far from the arc simply stops mattering.
2. **Keep the invariance check.** With the perturber's mass parameter set to
   zero, the perturbed arm must reproduce the null arm's log-likelihood exactly.
   That equality is what proves the two arms differ only by the perturber.
3. **Run the two model-independent diagnostics before quoting `delta ln Z`:**
   * the **nesting bound** `ln Z(wide) >= ln Z(narrow) - n_freed ln f`, with the
     null arm's measured response as the calibration for what widening should do
     when nothing is clipped (a result *below* the bound means the sampler
     failed, not that the perturber vanished);
   * the **edge-tracking** test of §3.2 (posterior median distance to the prior
     edge in posterior sigma, at two or three widths).
4. **Report the position posterior's radial distribution**, not only its median
   pair. A wide position prior can return a confident-looking `delta ln Z` while
   containing none of the known mode.
5. **If the drift persists after the profile is truncated**, stop varying the
   perturber and look at the smooth-model family instead (source
   regularisation strength, source template freedom, whether the smooth model's
   own posterior is being clipped). The truncation removes the free
   constant/gradient channel; whatever remains is a different problem.
6. **Per-cell (tiled) searches do not remove this degeneracy.** Partitioning the
   plane and running a perturber per tile constrains the *position*, which is the
   part of the problem that §3.5 shows is already recoverable; the mass
   degeneracy of §2 lives on the mass side of the trade and is unaffected. Always
   combine tiles by the prior-partition identity
   `Z_total = sum_i (A_i / A_total) Z_i`, never by taking `max_i delta ln Z`.

Expected outcome after switching to a truncated profile, stated in advance so
the fix can be judged: (a) `delta ln Z` converges across the 5/10/20
`sigma_M3` floors instead of 26 -> 48 -> 82; (b) the perturbed arm's `theta_E`
posterior no longer hugs the prior edge, and the drift stops scaling with the
floor; (c) the recovered `theta_E_sub` is stable across floors.

### 4.1 Adopted model choice: Einstein-radius tidal approximation

The current J0946 pipeline deliberately uses a **Vegetti-style singular
Pseudo-Jaffe subhalo, not an exact reproduction of Vegetti et al. (2010)**.
It retains the circular, zero-core, smoothly truncated profile from
[their equation (2), section 5.2](https://doi.org/10.1111/j.1365-2966.2010.16865.x):

\[
\kappa(R)=\frac{b_{\rm sub}}{2}
\left[\frac{1}{R}-\frac{1}{\sqrt{R^2+r_t^2}}\right].
\]

Our adopted tidal-radius prescription is

\[
r_t=\sqrt{b_{\rm sub}\,\theta_{E,\rm host}},
\]

where `theta_E_host` is the sampled main-lens Einstein-radius parameter, in the
same angular units as `b_sub` and `r_t`. It is intentionally used as the host
scale instead of converting the EPL parameters to the paper's main-lens density
normalization `b`. Those quantities are not generally equal for a
non-isothermal EPL.

This choice keeps a finite-mass subhalo tied to the live host Einstein radius
without adding an independent truncation-radius parameter. The square-root
relation is an approximation adopted for this model, not a general
tidal-stripping calculation. The radial profile follows Vegetti's convention;
the host-strength prescription does not reproduce the paper exactly. The
pipeline's inference priors should likewise not be assumed to reproduce the
paper merely because the profile is described as "Vegetti-style".

The numerical prescription is unchanged by recording this decision. The SIS
experiments above remain historical diagnostics, not validation of the new
Pseudo-Jaffe model's evidence or prior-width convergence.

### 4.2 Configuration: globals only

M4 and M5 now read the module-level settings directly. Both use the same
`build_comparison_likelihood` function and the same M2/M3 posteriors: M3 sets
the smooth-mass top-hats, and M2 sets the regularization centre and fixed `rho`.
There is no frozen comparison object or `comparison_prior.json` output.
Keep the global settings unchanged between the two arms of a comparison.

The sampler's `prior_specs` in each stage pickle record the priors actually
used; they are output records, not a second configuration source. The M4/M5
cache key includes the derived priors, position penalty, source grid, pixel
scale, subsampling, regularization, solver and sampler settings. It also hashes
the lens-subtracted image, noise, PSF, combined mask and fixed source template
and bbox. Changing any of these inputs invalidates both comparison caches;
the new pipeline schema rejects caches from older versions.
This check applies to M4/M5 only: `--skip-done` can still reuse earlier A–M3
stages after changes to *their* inputs. Use a fresh output directory when
changing upstream settings or data.
The Einstein-radius tidal approximation in §4.1 is unchanged.

`SUBHALO_POSITION_MODE` is read when the model is built, and
`LAMBDA_PRIOR_HALF_WIDTH_DEX` is converted to natural-log units at that time.
`COMPARISON_USE_POSITION_LIKELIHOOD` directly controls the position penalty in
both arms. No JSON configuration or saved prior record overrides these globals.

## 5. Context for this system

J0946+1006 is a multi-source-plane lens, which makes a wide position prior
especially dangerous: there is a candidate screen at each ring radius, so a
perturber that is free to move can pick a different ring rather than the known
one. The literature reports the inner ring at `R_E1 = 1.43 +- 0.01` arcsec
(`z_s1 = 0.609`; Gavazzi et al. 2008) and the outer ring at ~2.1 arcsec, with a
third source at ~2.5 arcsec (Collett & Smith 2020); the subhalo detection itself
is Vegetti et al. (2010), with several independent re-analyses since, and the
prevalence of alternatives in that literature is itself a reason to want the
degeneracy handled explicitly rather than absorbed into the perturber's mass.

Using a *truncated* perturber and marginalising over its parameters is what the
gravitational-imaging literature does; the practical lesson here is that this is
not a stylistic choice. With an untruncated profile the comparison is not merely
less sensitive — it is unbounded, and the resulting `delta ln Z` is a function of
the shared prior's width rather than of the data.

## 6. Reproduction

Everything below ran on one NVIDIA GeForce RTX 5080 (16 GB) with
`XLA_PYTHON_CLIENT_PREALLOCATE=false`, float32. At `n_live = 1000` a full M4/M5
pair took 8.8 min at the 5 `sigma_M3` floor (193.5 s + 332.0 s), 10.6 min at 10,
and 13.1 min at 20 (`time_taken` recorded in each stage payload); the wider
floors cost more because the perturbed arm's sampling efficiency drops.

These timings and evidence values describe the historical SIS implementation.
The commands below now run the Pseudo-Jaffe model and must not be interpreted
as reproducing the old numerical results. Select the desired position mode in
the module globals before running; the current default is `gaussian_local`.

```bash
cd examples/pix_src_demo_operator/pipe/subhalo_0946
# 5 sigma floor; set SUBHALO_POSITION_MODE = "full_image" first for a full-image run
python model.py --skip-done --out-dir output_m4m5_v3
# localised position prior (truncated Gaussian, sigma = 0.2 arcsec)
python model.py --skip-done --out-dir output_m4m5_local \
    --subhalo-position -0.70744 1.00661 --subhalo-position-sigma 0.2
# width sensitivity, both configurations
python model.py --skip-done --out-dir output_m4m5_local_s10 --smooth-sigma-floor 10
python model.py --skip-done --out-dir output_m4m5_local_s20 --smooth-sigma-floor 20
```

The position prior is selected with `SUBHALO_POSITION_MODE`
(`full_image` | `model_square` | `gaussian_local`). Its current bounds and
distribution enter the comparison cache key together with the other priors and
the position-penalty switch.

The cached-result probes below consume newly generated Pseudo-Jaffe posteriors.
`probe_sis_farfield.py` remains a standalone historical SIS calculation.
`smoke_m4_m5.py` uses synthetic data and needs no sampler or stage caches.
The PJ strength is `b_subhalo` (arcsec), not the former SIS `theta_E_subhalo`;
old SIS caches must be regenerated, not relabeled. The truncation radius follows
`rt = sqrt(b_subhalo * theta_E_host)` using the intentional Einstein-radius
approximation. The decomposition probe compares PJ-off and M4 at identical
shared parameters; its likelihood differences are not a decomposition of
`delta log Z`.

For a cache-free numerical check, run from the repository root:

```bash
python .scratch/subhalo-prior-audit/smoke_m4_m5.py
```

This exercises the actual likelihood with synthetic data, checks the PJ-off
limit at both the original and shifted shared-parameter points, and verifies
the pointwise likelihood decomposition. It does not perform nested sampling.


| Probe | Question |
| --- | --- |
| `.scratch/subhalo-prior-audit/probe_sis_farfield.py` | What fraction of the perturber's deflection field over the arc is expressible as `{constant + gradient}`? (§2) |
| `.scratch/subhalo-prior-audit/probe_position_modes.py` | Radial distribution and 2D peaks of the stored position posterior (§3.4) |
| `.scratch/subhalo-prior-audit/verify_lm_decomposition.py` | Splits `delta ln L` into "perturber present" vs "smooth parameters moved" (§3.3) |
| `.scratch/subhalo-prior-audit/probe_theta_e_slide.py` | Is the perturbed arm's optimum inside the shared prior or clipped outside it? |
| `.scratch/subhalo-prior-audit/probe_ridge_chase.py` | Does a finite joint optimum exist, or is there a ladder of ever-better optima? |
| `.scratch/subhalo-prior-audit/inspect_local_m5.py` | Weighted PJ-strength quantiles and cumulative weight below thresholds |
| `.scratch/subhalo-prior-audit/inspect_m5_modal_structure.py` | Posterior shape and boundary weight against saved prior records |
| `.scratch/subhalo-prior-audit/probe_subhalo_effect.py` | PJ deflection/Jacobian changes and optional `b_subhalo` likelihood scan |
| `.scratch/subhalo-prior-audit/probe_position_penalty.py` | Source-plane position penalty with PJ strength and position varied |
| `.scratch/subhalo-prior-audit/smoke_m4_m5.py` | Shared priors, PJ on/off invariance and pointwise likelihood decomposition |

## 7. Limitations

* This is a modeling lesson with one worked example (SLACS J0946+1006) and one
  pipeline configuration (pixelized source with a fixed template, regularisation
  strength taken from a calibration stage, `adaptive_reg_rho` fixed). The
  degeneracy argument of §1-2 is profile-level and configuration-independent;
  the sizes of the evidence shifts are not.
* No detection claim is made either way. The comparison is an
  empirical-Bayes/conditional one, and the numbers in §3 are quoted only as
  evidence that the statistic is prior-volume driven.
* The external numbers in §5 (ring radii, source redshifts, previous detection)
  are quoted from the abstracts and analyses cited, not re-measured here.
* The `{constant + gradient}` projection measures a *first-order* proxy of what
  the main lens's parameters can generate over the ring, which is why the
  on-ring value is 85% rather than 100%. The trend with radius — increasing
  monotonically towards unity — is the robust part.
