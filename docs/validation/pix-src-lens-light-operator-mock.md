# Joint operator mock validation

Validation date: 2026-07-11

The mock and fit were run from
`examples/pix_src_demo_operator/simple/pix_src_lens_light` in the
`tinylens_gpu` environment:

```bash
python sim_data.py
python fit_joint.py
```

The fit used a fixed 120 x 120 source grid, PCG joint inversion, a Gaussian
source-plane position likelihood, Nautilus `n_live=300`, sampler
`n_batch=200`, and internal JAX likelihood chunks of 50.

## Posterior recovery

| Parameter | Truth | Median | 16th--84th percentile |
|---|---:|---:|---:|
| theta_E | 1.0 | 1.000231 | 0.999808--1.000638 |
| e1 | 0.1 | 0.098994 | 0.098133--0.099909 |
| e2 | 0.0 | 0.000900 | -0.000008--0.001852 |
| center_x | 0.0 | 0.000396 | -0.000935--0.001738 |
| center_y | 0.0 | -0.000473 | -0.001494--0.000526 |

For this noise realization, all mass-parameter absolute offsets are at most
0.00101. The `e1` truth is about 1.1 posterior standard deviations from the
median; the other four truths lie inside their central 68% intervals. This
supports no significant mass bias for this mock, rather than an ensemble
coverage claim.

## Residual diagnostics

The reconstructed lensed-source signal-to-noise greater than 3 defines the
arc region.

| Metric | Value |
|---|---:|
| chi2 / Ndata | 0.93896 |
| Arc normalized-residual mean | 0.02020 |
| Arc normalized-residual standard deviation | 0.94078 |
| Arc residual--template correlation | 0.01453 |

The normalized residual image is noise-like, with no visible ring or arc
structure. Nautilus completed 36,400 likelihood calls in 466.2 seconds with
no accelerator out-of-memory error; the final log evidence was 2862.51.

As a deterministic-bias check, five antithetic noise pairs were also scanned
locally at fixed nuisance parameters. The pair-averaged offsets were
`(+0.000071, +0.000288, -0.000454, -0.000638, -0.000102)` for
`(theta_E, e1, e2, center_x, center_y)`, respectively. These offsets are all
smaller than the corresponding single-realization posterior uncertainty.
