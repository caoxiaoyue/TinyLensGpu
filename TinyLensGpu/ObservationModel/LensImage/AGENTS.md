# OBSERVATION MODEL KB

## OVERVIEW
This directory is the likelihood / evidence bridge over simulators. It turns forward-model images into chi-square, multi-band, pixelized-evidence, and point-source position likelihoods.

## STRUCTURE
```text
TinyLensGpu/ObservationModel/LensImage/
|- parametric_image_model.py    # ImageProbModel (chi-square image likelihood)
|- multi_band_image_model.py    # MultiBandImageProbModel + BandImageData
|- pixelized_image_model.py     # PixelizedImageProbModel (source-inversion evidence)
|- point_source_model.py        # PointSourceProbModel (position likelihood)
`- __init__.py                  # exports 5 public symbols
```

## WHERE TO LOOK
| Task | Location | Notes |
|------|----------|-------|
| Image chi-square likelihood | `TinyLensGpu/ObservationModel/LensImage/parametric_image_model.py` | Primary demo-facing likelihood |
| Multi-band image likelihood | `TinyLensGpu/ObservationModel/LensImage/multi_band_image_model.py` | Band geometry + per-band metadata |
| Pixelized source evidence | `TinyLensGpu/ObservationModel/LensImage/pixelized_image_model.py` | Regularized linear inversion + evidence |
| Point-source position likelihood | `TinyLensGpu/ObservationModel/LensImage/point_source_model.py` | Lens-equation solving + permutation matching |

## CONVENTIONS
- All likelihood classes are `ck.Module` subclasses with a `__call__` method returning scalar log-likelihood.
- `ImageProbModel` is the canonical parametric-image likelihood; demos usually build this directly.
- `PixelizedImageProbModel` uses dense regularization builders from `utils.inversion.regularization`; keep the regularization type string normalized before branching.
- Multi-band models accept a list of `BandImageData` dataclasses; band-to-band shifts/rotations are relative to the reference band.
- Point-source likelihood uses permutation-invariant matching; brute-force for small multiplicities, Hungarian for large.

## ANTI-PATTERNS
- Do not bypass `ImageProbModel` callable expectations when writing custom likelihood wrappers; samplers expect `ck.Module`-like `__call__` semantics.
- Do not manually construct pixelized regularization matrices outside the `DenseRegularizationBuilder` helpers; the builder owns covariance structure and prior normalization.
- Do not ignore the `position_likelihood` config dict shape; it must contain `positions`, `threshold_arcsec`, and `min_log_like` keys when used.

## NOTES
- `LensLikelihood` is a backward-compatible alias for `ImageProbModel`.
- `__init__.py` exports 5 symbols; new likelihood classes should be added there.
