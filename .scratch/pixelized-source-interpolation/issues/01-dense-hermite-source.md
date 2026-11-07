# 01: Reconstruct an opt-in Hermite source with the dense backend

**What to build:** A lens modeler can explicitly select `cubic_hermite` on a `PixelizedSourceModel`, evaluate source brightness, and reconstruct a source through the dense backend. The fit uses the original node intensities and the same shared weights as direct brightness evaluation. Existing calls continue to select bilinear by default.

**Blocked by:** None (can start immediately).

**Status:** ready-for-agent
**State:** open

**Spec:** [Parent specification](../spec.md)

- [ ] Source construction validates the two static interpolation modes. Omitted and explicit bilinear settings agree with the existing dense reconstruction and direct mapping behavior, including hard-box clipping and the upper-edge convention. Existing explicit bilinear callers continue to work.
- [ ] The shared mapping seam returns weights, original-node indices, and geometric support for either mode. Hermite uses the experimental centered-difference tensor product and directly assembles source columns for the original `n²` nodes without a padded array or matrix.
- [ ] An independent zero-extended Hermite reference verifies nodal reproduction, signed weights, the smallest supported grid, and nonzero contributions to rays outside the fitted box but inside the added support.
- [ ] Spatial derivatives use the same interpolation definition as brightness and agree with automatic differentiation and finite differences. Check continuity at internal knots, fitted edges, and virtual lines, including a zero-weight row with a nonzero normal derivative. Brightness and first derivatives vanish at and beyond the outer support.
- [ ] Direct brightness and a dense reconstruction agree on the selected mapping for a small case with a mask, asymmetric PSF, and subpixel integration. Exercise fixed and freshly prepared geometry and preserve the ray coordinate precision.

