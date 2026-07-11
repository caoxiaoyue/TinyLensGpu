# Joint operator mock validation

Validation date: 2026-07-12

The mock and fit were run from
`examples/pix_src_demo_operator/simple/pix_src_lens_light` in the
`tinylens_gpu` environment:

```bash
python sim_data.py
python fit_joint.py
```

The current fit used an 80 x 80 source grid whose bbox is re-inferred for each
mass model, PCG joint inversion, the original threshold-based source-plane
position likelihood (`threshold_arcsec=0.3`, `min_log_like=-1e10`), Nautilus
`n_live=300`, sampler `n_batch=200`, and internal JAX likelihood chunks of 50.

## Posterior recovery

| Parameter | Truth | Median | 16th--84th percentile |
|---|---:|---:|---:|
| theta_E | 1.0 | 0.999373 | 0.999017--0.999722 |
| e1 | 0.1 | 0.096978 | 0.095966--0.098018 |
| e2 | 0.0 | 0.000571 | -0.000355--0.001495 |
| center_x | 0.0 | -0.001924 | [-0.003472, -0.000276] |
| center_y | 0.0 | -0.003582 | [-0.004788, -0.002340] |

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
| chi2 / Ndata | 0.94049 |
| Arc normalized-residual mean | 0.01736 |
| Arc normalized-residual standard deviation | 0.94147 |
| Arc residual--template correlation | 0.01633 |

The normalized residual image is noise-like, with no visible ring or arc
structure. Nautilus completed 35,600 likelihood calls in 263.7 seconds with
no accelerator out-of-memory error; the final log evidence was 2896.52.

## 80 versus 120 dynamic grids

Changing only the source-grid resolution produced:

| Quantity | 120 x 120 | 80 x 80 |
|---|---:|---:|
| theta_E | 0.999065 | 0.999373 |
| e1 | 0.096733 | 0.096978 |
| e2 | 0.000412 | 0.000571 |
| center_x | -0.001133 | -0.001924 |
| center_y | -0.003468 | -0.003582 |
| chi2 / Ndata | 0.93883 | 0.94049 |
| Arc residual--template correlation | 0.01949 | 0.01633 |
| log evidence approximation | 2868.00 | 2896.52 |
| Runtime (seconds) | 444.8 | 263.7 |

Both resolutions leave noise-like arc residuals. The 80 x 80 run is about 41%
faster; its mass recovery is mixed rather than uniformly better or worse.
Evidence values should not be compared as though the linear source dimension
were unchanged, because the operator determinant is an approximation whose
normalization changes with grid resolution.

## Fixed versus dynamic bbox at 120 x 120

Changing only the bbox policy produced the following comparison:

| Quantity | Fixed bbox | Dynamic bbox |
|---|---:|---:|
| theta_E | 0.999472 | 0.999065 |
| e1 | 0.097337 | 0.096733 |
| e2 | 0.000441 | 0.000412 |
| center_x | -0.001214 | -0.001133 |
| center_y | -0.003379 | -0.003468 |
| chi2 / Ndata | 0.93788 | 0.93883 |
| Arc residual--template correlation | 0.01712 | 0.01949 |
| log evidence | 2869.33 | 2868.00 |

The arc reconstruction is effectively unchanged, while the dynamic bbox moves
`theta_E` and `e1` farther from their truths and slightly lowers the evidence.

As a deterministic-bias check, five antithetic noise pairs were also scanned
locally at fixed nuisance parameters. The pair-averaged offsets were
`(+0.000071, +0.000288, -0.000454, -0.000638, -0.000102)` for
`(theta_E, e1, e2, center_x, center_y)`, respectively. These offsets are all
smaller than the corresponding single-realization posterior uncertainty.
