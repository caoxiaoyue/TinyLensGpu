# INVERSION KB

## OVERVIEW
This directory houses dense regularization builders for pixelized source inversions. It constructs covariance matrices and precision operators used by `PixelizedImageProbModel`.

## STRUCTURE
```text
TinyLensGpu/utils/inversion/
|- regularization.py    # DenseRegularizationBuilder + GP kernel factories
`- __init__.py          # package exports
```

## WHERE TO LOOK
| Task | Location | Notes |
|------|----------|-------|
| Regularization matrix construction | `TinyLensGpu/utils/inversion/regularization.py` | First/second-order finite difference + GP kernels |
| Cholesky factorization for evidence | `TinyLensGpu/utils/inversion/regularization.py` | Single Cholesky for precision + logdet |

## CONVENTIONS
- Regularization type strings are normalized with lowercase/strip before branching.
- `DenseRegularizationBuilder` owns covariance structure, prior normalization, and log-determinant computation.
- Supported GP kernels: `matern32`, `matern52`, `matern72`, `exponential`, `gaussian`.

## ANTI-PATTERNS
- Do not manually build regularization matrices outside the builder; the builder handles covariance structure and normalization consistently.
- Do not remove numerical safeguards (jitter, floors) in the Cholesky path; they stabilize the evidence calculation.

## NOTES
- `regularization.py` is one of the highest-risk numerical cores in the package.
