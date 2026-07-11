# Joint operator mock validation

Validation date: 2026-07-12

The mock and fit were run from
`examples/pix_src_demo_operator/simple/pix_src_lens_light` in the
`tinylens_gpu` environment:

```bash
python sim_data.py
python fit_joint.py
```

The fit used a fixed 120 x 120 source grid, PCG joint inversion, the original
threshold-based source-plane position likelihood (`threshold_arcsec=0.3`,
`min_log_like=-1e10`), Nautilus `n_live=300`, sampler `n_batch=200`, and
internal JAX likelihood chunks of 50.

## Posterior recovery

| Parameter | Truth | Median | 16th--84th percentile |
|---|---:|---:|---:|
| theta_E | 1.0 | 0.999472 | 0.999057--0.999939 |
| e1 | 0.1 | 0.097337 | 0.096279--0.098350 |
| e2 | 0.0 | 0.000441 | -0.000404--0.001262 |
| center_x | 0.0 | -0.001214 | -0.002478--0.000025 |
| center_y | 0.0 | -0.003379 | [-0.004496, -0.002238] |

After restoring the original position likelihood, this noise realization again
shows the previously measured `e1` and `center_y` posterior shifts. This is
recorded explicitly rather than attributing their improvement to the source
inversion changes. The arc residual quality below remains essentially
unchanged, demonstrating that the Gaussian position term was not responsible
for removing the structured arc residual.

## Residual diagnostics

The reconstructed lensed-source signal-to-noise greater than 3 defines the
arc region.

| Metric | Value |
|---|---:|
| chi2 / Ndata | 0.93788 |
| Arc normalized-residual mean | 0.01903 |
| Arc normalized-residual standard deviation | 0.93868 |
| Arc residual--template correlation | 0.01712 |

The normalized residual image is noise-like, with no visible ring or arc
structure. Nautilus completed 35,600 likelihood calls in 454.4 seconds with
no accelerator out-of-memory error; the final log evidence was 2869.33.

As a deterministic-bias check, five antithetic noise pairs were also scanned
locally at fixed nuisance parameters. The pair-averaged offsets were
`(+0.000071, +0.000288, -0.000454, -0.000638, -0.000102)` for
`(theta_E, e1, e2, center_x, center_y)`, respectively. These offsets are all
smaller than the corresponding single-realization posterior uncertainty.
