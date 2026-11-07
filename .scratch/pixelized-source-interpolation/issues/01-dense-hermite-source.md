# 01: Reconstruct an opt-in Hermite source with the dense backend

**What to build:** A lens modeler can explicitly select `cubic_hermite` on a `PixelizedSourceModel`, evaluate source brightness, and reconstruct a source through the dense backend. The fit uses the original node intensities and the same shared weights as direct brightness evaluation. Existing calls continue to select bilinear by default.

**Blocked by:** None (can start immediately).

**Status:** ready-for-agent
**State:** resolved

**Spec:** [Parent specification](../spec.md)

- [x] Source construction validates the two static interpolation modes. Omitted and explicit bilinear settings agree with the existing dense reconstruction and direct mapping behavior, including hard-box clipping and the upper-edge convention. Existing explicit bilinear callers continue to work.
- [x] The shared mapping seam returns weights, original-node indices, and geometric support for either mode. Hermite uses the experimental centered-difference tensor product and directly assembles source columns for the original `n²` nodes without a padded array or matrix.
- [x] An independent zero-extended Hermite reference verifies nodal reproduction, signed weights, the smallest supported grid, and nonzero contributions to rays outside the fitted box but inside the added support.
- [x] Spatial derivatives use the same interpolation definition as brightness and agree with automatic differentiation and finite differences. Check continuity at internal knots, fitted edges, and virtual lines, including a zero-weight row with a nonzero normal derivative. Brightness and first derivatives vanish at and beyond the outer support.
- [x] Direct brightness and a dense reconstruction agree on the selected mapping for a small case with a mask, asymmetric PSF, and subpixel integration. Exercise fixed and freshly prepared geometry and preserve the ray coordinate precision.

## Comments

Claimed for implementation on `codex/pixelized-source-interpolation`.

Implemented in `0ffbfcd`, merged as `db40705`. In `tinylens_gpu` from `tests/`, the
9 interpolation checks passed on CPU. With `JAX_PLATFORMS=cuda` and
`XLA_PYTHON_CLIENT_PREALLOCATE=false`, the following command passed 181 tests.

```bash
pytest -xq test_pixelized_source_interpolation.py test_pixelized_source_model.py   test_pixelized_source_utils.py test_pixelized_inversion.py test_image_template_light.py   'test_pixelized_source_precision.py::test_reconstruction_keeps_explicit_double_precision_source_geometry[dense]'
```
