# TinyLensGpu Lens Modeling

This context defines the modeling language used for strong-lens image reconstruction and inference.

## Language

**Singular Isothermal Sphere (SIS)**:
A circularly symmetric singular isothermal mass profile parameterized by an Einstein radius and lens center, without ellipticity parameters. It is a first-class mass profile rather than an SIE configuration.
_Avoid_: Circular SIE, SIE wrapper

**SIE circular limit**:
The lensing behavior approached by a Singular Isothermal Ellipsoid as both ellipticity components tend to zero; it must agree with the corresponding SIS away from the shared central singularity.
_Avoid_: SIS approximation

**Subhalo mass component**:
An additional localized mass profile inferred alongside the main lens mass and external shear; in the subhalo-search pipeline it is represented by an SIS with its own center and Einstein radius.
_Avoid_: Main-lens SIS, circular main lens

**Joint semi-linear inversion**:
A single linear inversion that solves pixelized lensed-source intensities and parametric lens-light component intensities together, conditional on the nonlinear model parameters. Its regularization precision is block diagonal, with source regularization on the source pixels and weak zero-order regularization on the lens-light intensities.
_Avoid_: Lens-light subtraction, separate source/lens-light fitting

**Lens-light intensity**:
The linear amplitude of one unit-amplitude parametric lens-light basis component in a joint semi-linear inversion.
_Avoid_: Lens-light pixel, lens-light nonlinear parameter

**Unit-amplitude lens-light basis**:
A parametric lens-light component whose sole internal intensity parameter is static and equal to one, leaving its fitted lens-light intensity as the only amplitude scale.
_Avoid_: Dynamically normalized lens-light component, doubly scaled lens-light component

**Robust mixture prior**:
A Bayesian inference prior that combines a bounded informative core with a broad component on the same physical support, preserving nonzero prior mass away from the core estimate. It changes the posterior and evidence measure rather than acting only as a sampling proposal.
_Avoid_: Search hint, proposal distribution, physical-model improvement
