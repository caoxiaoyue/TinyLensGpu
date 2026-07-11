# TinyLensGpu Lens Modeling

This context defines the modeling language used for strong-lens image reconstruction and inference.

## Language

**Joint semi-linear inversion**:
A single linear inversion that solves pixelized lensed-source intensities and parametric lens-light component intensities together, conditional on the nonlinear model parameters. Its regularization precision is block diagonal, with source regularization on the source pixels and weak zero-order regularization on the lens-light intensities.
_Avoid_: Lens-light subtraction, separate source/lens-light fitting

**Lens-light intensity**:
The linear amplitude of one unit-amplitude parametric lens-light basis component in a joint semi-linear inversion.
_Avoid_: Lens-light pixel, lens-light nonlinear parameter

**Unit-amplitude lens-light basis**:
A parametric lens-light component whose sole internal intensity parameter is static and equal to one, leaving its fitted lens-light intensity as the only amplitude scale.
_Avoid_: Dynamically normalized lens-light component, doubly scaled lens-light component
