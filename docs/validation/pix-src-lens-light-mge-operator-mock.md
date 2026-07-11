# Joint MGE operator mock validation

Validation date: 2026-07-12

The fit was run from
`examples/pix_src_demo_operator/simple/pix_src_lens_light_mge` with:

- an 80 x 80 dynamic source grid and `source_bbox_padding=0.2`;
- ten fixed-width Gaussian lens-light bases with jointly solved non-negative
  amplitudes;
- a block-Schur PCG warm start followed by 500 FISTA iterations at
  `rtol=1e-3`;
- `nsub=4`, Nautilus `n_live=300`, sampler `n_batch=200`, and internal JAX
  chunks of 50.

Nautilus completed 28,200 likelihood calls in 814.6 seconds without an
accelerator out-of-memory error. The final log evidence approximation was
2827.59.

## Posterior recovery

| Parameter | Truth | Median | 16th--84th percentile |
|---|---:|---:|---:|
| theta_E | 1.0 | 0.999674 | 0.999242--1.000174 |
| e1 | 0.1 | 0.097396 | 0.096254--0.098758 |
| e2 | 0.0 | 0.000760 | -0.000202--0.001684 |
| center_x | 0.0 | -0.001586 | -0.003186--0.000146 |
| center_y | 0.0 | -0.003956 | [-0.005365, -0.002489] |

## Residual diagnostics

| Metric | Value |
|---|---:|
| chi2 / Ndata | 0.94079 |
| Arc normalized-residual mean | 0.01691 |
| Arc normalized-residual standard deviation | 0.94258 |
| Arc residual--template correlation | 0.02011 |

The residual image is noise-like with no visible coherent lensed-arc pattern.
At the posterior median, all source pixels and MGE amplitudes remain
non-negative. A truth-point convergence comparison gave log-evidence values
2864.832, 2864.878, and 2864.890 for 500, 1000, and 5000 FISTA iterations,
respectively, so the production 500-iteration result differs from the
high-iteration reference by about 0.06 in log likelihood.

The original 20-component MGE did not converge even after 5000 zero-start or
PCG-warm-started FISTA iterations. Ten components were sufficient for this
single-Sersic mock and removed the strongly redundant basis directions.
