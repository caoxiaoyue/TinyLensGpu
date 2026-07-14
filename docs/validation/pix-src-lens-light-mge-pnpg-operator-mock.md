# Joint 20-MGE PNPG operator validation

Validation date: 2026-07-14

The fit was run on an NVIDIA GeForce RTX 5080 from
`examples/pix_src_demo_operator/simple/pix_src_lens_light_mge/fit_joint.py`
with:

- an 80 x 80 dynamic source grid and `nsub=4`;
- twenty fixed-width Gaussian lens-light bases with jointly solved
  non-negative amplitudes;
- `solver_type="pnpg"`, 1000 projected-Nesterov iterations, a componentwise
  KKT tolerance of `2e-2`, and 20 power iterations;
- Nautilus `n_live=125`, `n_batch=100`, `n_eff=400`, `f_live=0.1`, and
  likelihood chunks of 50.

The final timed script completed 13,800 likelihood calls with 869.7 seconds of
sampling and a total wall time of 883.4 seconds (14.72 minutes). It produced
`N_eff=1769` with `log(Z)=2769.58`. An independent repeat of the same target
configuration completed 13,500 calls in 852.7 sampling seconds and gave
`log(Z)=2769.49` with `N_eff=1696`.

## Posterior recovery

| Parameter | Truth | Median | 16th--84th percentile |
|---|---:|---:|---:|
| theta_E | 1.0 | 0.9994 | 0.9989--0.9999 |
| e1 | 0.1 | 0.0973 | 0.0961--0.0984 |
| e2 | 0.0 | 0.0006 | -0.0003--0.0015 |
| center_x | 0.0 | -0.0022 | -0.0036 to -0.0003 |
| center_y | 0.0 | -0.0033 | -0.0046 to -0.0019 |
| log_lambda_reg | -- | -7.9762 | -8.0277 to -7.9229 |

## Residual diagnostics

| Metric | Value |
|---|---:|
| chi2 / Ndata | 0.9402 |
| Arc normalized-residual mean | 0.0162 |
| Arc normalized-residual standard deviation | 0.9424 |
| Arc residual--template correlation | 0.0202 |

The two final target-configuration runs agree in log evidence to 0.09 and
return matching posterior intervals. The final residual is noise-like, and all
returned source pixels and MGE amplitudes satisfy the non-negative constraint.
