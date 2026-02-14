"""Configuration models for pixelized-source reconstruction.

This module defines dataclasses and constants used to configure the various aspects of 
pixelized source modeling, including grid geometry, source-to-image mapping, 
regularization schemes, and linear inversion solvers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional, Tuple, Union


# Valid kernel types for Gaussian Process (GP) based regularization.
# These kernels define the covariance structure of the source surface brightness.
REG_KERNEL_TYPES = {"exp", "gauss", "matern32", "matern52"}

# Supported sampling methods for irregular source-plane meshes.
# 'random' uses uniform random sampling, while 'sobol' uses low-discrepancy sequences.
MESH_METHODS = {"random", "sobol"}

# Supported interpolation kernels for mapping source to image plane.
# These are Wendland functions with different degrees of smoothness (C2, C4, C6).
INTERP_KERNELS = {"wendland_c2", "wendland_c4", "wendland_c6"}

# Available backends for solving the linear inversion problem.
# 'matrix' constructs the full matrix explicitly, 'operator' uses matrix-free linear operators.
SOLVER_BACKENDS = {"matrix", "operator"}

# Maps rectangular regularization keywords to their corresponding operator types.
# These operators represent different levels of smoothing for rectangular grids.
RECTANGULAR_SCHEME_TO_OPERATOR = {
    "rectangular_zero": "zero",        # L0 regularization (Identity): favors small amplitudes.
    "rectangular_first": "gradient",    # L1 regularization (1st-order): favors flat profiles.
    "rectangular_second": "curvature",  # L2 regularization (2nd-order): favors smooth profiles.
}

# Dynamically generate GP-based irregular regularization schemes.
# Format: 'irregular_gp_<kernel_type>' (e.g., 'irregular_gp_exp').
IRREGULAR_GP_SCHEME_TO_KERNEL = {f"irregular_gp_{kernel}": kernel for kernel in sorted(REG_KERNEL_TYPES)}

# Dynamically generate KNN-based irregular regularization schemes.
# Format: 'irregular_knn_<kernel_type>' (e.g., 'irregular_knn_gauss').
IRREGULAR_KNN_SCHEME_TO_KERNEL = {f"irregular_knn_{kernel}": kernel for kernel in sorted(REG_KERNEL_TYPES)}

# Combined set of all valid regularization scheme identifiers used for validation.
REGULARIZATION_SCHEMES = (
    set(RECTANGULAR_SCHEME_TO_OPERATOR)
    | set(IRREGULAR_GP_SCHEME_TO_KERNEL)
    | set(IRREGULAR_KNN_SCHEME_TO_KERNEL)
)

@dataclass(frozen=True)
class IrregularGridConfig:
    """
    Configuration for irregular source-plane mesh sampling.

    This class defines how an unstructured mesh is generated in the source plane,
    typically used for adaptive or non-rectangular source reconstructions.

    Attributes:
        n_source_points (int): Total number of source-plane mesh points. 
            Default is 1500.
        mesh_alpha (float): Density parameter for adaptive mesh sampling.
            Values > 0 favor regions with higher magnification or light intensity.
            Set to 0 for uniform sampling in ray-traced coordinates. Default is 0.0.
        mesh_blur_sigma (float): Smoothing scale (in arcsec) for mesh density estimation.
            Used to regularize the density map before sampling. Default is 0.0.
        mesh_method (Literal["random", "sobol"]): Sampling strategy.
            'random' provides stochastic points; 'sobol' uses quasi-random sequences
            for better spatial coverage. Default is 'random'.
        mesh_seed (int): Random seed for reproducible sampling results. Default is 42.
    """

    n_source_points: int = 1500
    mesh_alpha: float = 0.0
    mesh_blur_sigma: float = 0.0
    mesh_method: Literal["random", "sobol"] = "random"
    mesh_seed: int = 42

    def __post_init__(self) -> None:
        """
        Validate configuration parameters after initialization.
        Ensures all numerical values are within physical or logical bounds.
        """
        # Ensure number of source points is a positive integer
        if int(self.n_source_points) <= 0:
            raise ValueError(f"n_source_points must be positive, got {self.n_source_points}.")
        
        # Blur sigma must be non-negative
        if float(self.mesh_blur_sigma) < 0.0:
            raise ValueError(f"mesh_blur_sigma must be >= 0, got {self.mesh_blur_sigma}.")
        
        # Check if the sampling method is supported
        if str(self.mesh_method).strip().lower() not in MESH_METHODS:
            raise ValueError(
                f"Unknown mesh_method: '{self.mesh_method}'. Must be one of {MESH_METHODS}."
            )


@dataclass(frozen=True)
class RectangularGridConfig:
    """
    Configuration for rectangular (Cartesian) source-plane grids.

    This class defines the geometry and resolution of a regular grid used for
    standard pixelized source reconstructions.

    Attributes:
        nx (int): Number of pixels along the x-axis. Default is 64.
        ny (int): Number of pixels along the y-axis. Default is 64.
        margin_frac (float): Fraction of extra padding added around the 
            automatically determined ray-traced image extent. Default is 0.10.
        bounds (Optional[Tuple[float, float, float, float]]): Fixed coordinate boundaries 
            as (x_min, x_max, y_min, y_max) in arcsec. If None, boundaries are 
            derived from ray-tracing. Default is None.
    """

    nx: int = 64
    ny: int = 64
    margin_frac: float = 0.10
    bounds: Optional[Tuple[float, float, float, float]] = None

    def __post_init__(self) -> None:
        """
        Validate grid dimensions and coordinate boundaries.
        Ensures the grid has a valid shape and correctly ordered bounds.
        """
        # Pixel counts must be positive integers
        if int(self.nx) <= 0 or int(self.ny) <= 0:
            raise ValueError(f"Rectangular grid shape must be positive, got nx={self.nx}, ny={self.ny}.")
        
        # Margin fraction must be non-negative
        if float(self.margin_frac) < 0.0:
            raise ValueError(f"margin_frac must be >= 0, got {self.margin_frac}.")
        
        # Validate fixed bounds if provided
        if self.bounds is not None:
            if len(self.bounds) != 4:
                raise ValueError("bounds must be a 4-tuple: (x_min, x_max, y_min, y_max).")
            
            x_min, x_max, y_min, y_max = [float(v) for v in self.bounds]
            
            # Ensure boundaries are logical (max > min)
            if x_max <= x_min or y_max <= y_min:
                raise ValueError(
                    "bounds must satisfy x_max>x_min and y_max>y_min, got "
                    f"x:({x_min}, {x_max}), y:({y_min}, {y_max})."
                )


# Union type representing any supported grid configuration
GridConfig = Union[IrregularGridConfig, RectangularGridConfig]


@dataclass(frozen=True)
class MappingConfig:
    """
    Configuration for source-to-image interpolation mapping.

    This class controls how surface brightness from the source plane is 
    interpolated onto the image plane during ray-tracing.

    Attributes:
        k_neighbors (int): Number of nearest source pixels to use for interpolation 
            per image pixel. Higher values result in smoother but more 
            computationally expensive mapping. Default is 5.
        interp_kernel (Literal["wendland_c2", "wendland_c4", "wendland_c6"]): 
            Radial basis function (RBF) kernel type used for interpolation. 
            Wendland kernels are compactly supported. Default is 'wendland_c4'.
        radius_scale (float): Scaling factor for the interpolation kernel radius.
            Typically set such that the kernel covers the k_neighbors. Default is 1.5.
    """

    k_neighbors: int = 5
    interp_kernel: Literal["wendland_c2", "wendland_c4", "wendland_c6"] = "wendland_c4"
    radius_scale: float = 1.5

    def __post_init__(self) -> None:
        """
        Validate interpolation settings.
        Ensures kernels are supported and numerical parameters are positive.
        """
        # k_neighbors must be a positive integer
        if int(self.k_neighbors) <= 0:
            raise ValueError(f"k_neighbors must be positive, got {self.k_neighbors}.")
        
        # Check if the interpolation kernel is supported
        if str(self.interp_kernel).strip().lower() not in INTERP_KERNELS:
            raise ValueError(
                f"Unknown interp_kernel: '{self.interp_kernel}'. Must be one of {INTERP_KERNELS}."
            )
        
        # Radius scale must be strictly positive
        if float(self.radius_scale) <= 0.0:
            raise ValueError(f"radius_scale must be > 0, got {self.radius_scale}.")


@dataclass(frozen=True)
class RegularizationConfig:
    """
    Configuration for source regularization operators.

    This class specifies the constraints applied to the source surface brightness 
    to prevent over-fitting of noise and to ensure a smooth reconstruction.

    Attributes:
        scheme (str): String identifier for the regularization scheme.
            Must be one of the keys in RECTANGULAR_SCHEME_TO_OPERATOR, 
            IRREGULAR_GP_SCHEME_TO_KERNEL, or IRREGULAR_KNN_SCHEME_TO_KERNEL.
            Default is 'irregular_gp_exp'.
        sparse_k_neighbors (int): Number of neighbors used for sparse approximations 
            in GP or KNN based schemes. Helps in reducing the computational cost 
            of the regularization operator. Default is 16.
    """

    scheme: str = "irregular_gp_exp"
    sparse_k_neighbors: int = 16

    def __post_init__(self) -> None:
        """
        Ensure the selected scheme is supported and parameters are valid.
        Cross-checks against global registry of valid schemes.
        """
        # Validate that the scheme exists in our supported list
        if str(self.scheme).strip().lower() not in REGULARIZATION_SCHEMES:
            raise ValueError(
                f"Unknown regularization scheme: '{self.scheme}'. Must be one of {sorted(REGULARIZATION_SCHEMES)}."
            )
        
        # sparse_k_neighbors must be a positive integer
        if int(self.sparse_k_neighbors) <= 0:
            raise ValueError(f"sparse_k_neighbors must be positive, got {self.sparse_k_neighbors}.")

    @property
    def normalized_scheme(self) -> str:
        """
        Returns the scheme name in a standardized lowercase format.
        Useful for internal comparisons and dictionary lookups.
        """
        return str(self.scheme).strip().lower()

    @property
    def is_rectangular_scheme(self) -> bool:
        """
        True if the selected scheme is designed for rectangular Cartesian grids.
        """
        return self.normalized_scheme.startswith("rectangular_")

    @property
    def is_irregular_scheme(self) -> bool:
        """
        True if the selected scheme is designed for irregular unstructured meshes.
        """
        return self.normalized_scheme.startswith("irregular_")

    @property
    def mode(self) -> Literal["dense_gp", "sparse_knn", "sparse_rectangular"]:
        """
        Resolved high-level regularization mode derived from the scheme name.
        Determines the underlying mathematical approach (e.g., GP vs. FD).
        """
        scheme = self.normalized_scheme
        if scheme in RECTANGULAR_SCHEME_TO_OPERATOR:
            return "sparse_rectangular"
        if scheme in IRREGULAR_GP_SCHEME_TO_KERNEL:
            return "dense_gp"
        if scheme in IRREGULAR_KNN_SCHEME_TO_KERNEL:
            return "sparse_knn"
        
        # This should theoretically not be reached due to __post_init__ validation
        raise ValueError(
            f"Unknown regularization scheme: '{self.scheme}'. Must be one of {sorted(REGULARIZATION_SCHEMES)}."
        )

    @property
    def gp_kernel(self) -> Optional[Literal["exp", "gauss", "matern32", "matern52"]]:
        """
        Resolved GP kernel type for irregular schemes.
        Returns None if the scheme is rectangular (finite-difference based).
        """
        scheme = self.normalized_scheme
        
        # Check GP schemes first
        kernel = IRREGULAR_GP_SCHEME_TO_KERNEL.get(scheme)
        if kernel is not None:
            return kernel  # type: ignore[return-value]
        
        # Check KNN schemes which also use GP kernels for weights
        kernel = IRREGULAR_KNN_SCHEME_TO_KERNEL.get(scheme)
        if kernel is not None:
            return kernel  # type: ignore[return-value]
            
        return None

    @property
    def rect_scheme(self) -> Optional[Literal["zero", "gradient", "curvature"]]:
        """
        Resolved finite-difference operator type for rectangular schemes.
        Returns None if the scheme is irregular.
        """
        scheme = self.normalized_scheme
        rect = RECTANGULAR_SCHEME_TO_OPERATOR.get(scheme)
        if rect is not None:
            return rect  # type: ignore[return-value]
        return None

    def resolved_mode(self) -> Literal["dense_gp", "sparse_knn", "sparse_rectangular"]:
        """
        Alias for the 'mode' property. 
        Provides the high-level implementation strategy for the regularization.
        """
        return self.mode


@dataclass(frozen=True)
class SolverConfig:
    """
    Configuration for linear inversion backend behavior.

    This class defines how the linear system (Lens Matrix) is inverted to 
    solve for the source surface brightness and optionally lens light.

    Attributes:
        inversion_backend (Literal["matrix", "operator"]): Method used to solve 
            the linear system. 'matrix' uses explicit matrices (memory-intensive);
            'operator' uses matrix-free linear operators (computation-intensive).
            Default is 'matrix'.
        include_lens_light (bool): If True, includes lens light components 
            in the joint linear inversion. Default is False.
        nonnegative (bool): If True, uses non-negative least squares (NNLS) 
            algorithms to ensure physical (positive) source brightness. 
            Default is False.
        lens_light_ridge (float): Ridge regularization strength (Tikhonov) 
            specifically for lens light components. Default is 1e-8.
        cg_tol (float): Convergence tolerance for the Conjugate Gradient (CG) 
            iterative solver. Default is 1e-4.
        cg_maxiter (int): Maximum number of iterations for the CG solver. 
            Default is 120.
        slq_seed (int): Random seed for Stochastic Lanczos Quadrature (SLQ) 
            log-determinant estimation. Default is 0.
        slq_probes (int): Number of random probe vectors for SLQ estimation. 
            Default is 32.
        slq_steps (int): Number of Lanczos steps for each probe in SLQ. 
            Default is 60.
        operator_cache_policy (str): Caching strategy for operator-based backends 
            to reuse intermediate computations. Default is 'safe'.
        nnls_maxiter (int): Maximum iterations for the NNLS solver (e.g., FISTA). 
            Default is 600.
        nnls_tol (float): Convergence tolerance for the NNLS solver. 
            Default is 1e-6.
        nnls_lipschitz_iters (int): Number of power iterations for estimating 
            Lipschitz constants in NNLS solvers. Default is 12.
    """

    inversion_backend: Literal["matrix", "operator"] = "matrix"
    include_lens_light: bool = False
    nonnegative: bool = False
    lens_light_ridge: float = 1e-8

    # Iterative solver parameters (CG)
    cg_tol: float = 1e-4
    cg_maxiter: int = 120
    
    # Determinant estimation parameters (SLQ)
    slq_seed: int = 0
    slq_probes: int = 32
    slq_steps: int = 60
    operator_cache_policy: str = "safe"

    # Non-negative solver parameters (FISTA/NNLS)
    nnls_maxiter: int = 600
    nnls_tol: float = 1e-6
    nnls_lipschitz_iters: int = 12

    def __post_init__(self) -> None:
        """
        Validate all solver-related numerical parameters.
        Ensures convergence tolerances and iteration limits are valid.
        """
        # Validate inversion backend
        if str(self.inversion_backend).strip().lower() not in SOLVER_BACKENDS:
            raise ValueError(
                f"Unknown inversion_backend: '{self.inversion_backend}'. Must be one of {SOLVER_BACKENDS}."
            )
        
        # Ridge parameter must be non-negative
        if float(self.lens_light_ridge) < 0.0:
            raise ValueError(f"lens_light_ridge must be >= 0, got {self.lens_light_ridge}.")
        
        # CG parameters validation
        if float(self.cg_tol) <= 0.0:
            raise ValueError(f"cg_tol must be > 0, got {self.cg_tol}.")
        if int(self.cg_maxiter) <= 0:
            raise ValueError(f"cg_maxiter must be positive, got {self.cg_maxiter}.")
        
        # SLQ parameters validation
        if int(self.slq_probes) <= 0 or int(self.slq_steps) <= 0:
            raise ValueError(f"slq_probes and slq_steps must be positive, got {self.slq_probes}, {self.slq_steps}.")
        
        # NNLS parameters validation
        if int(self.nnls_maxiter) <= 0:
            raise ValueError(f"nnls_maxiter must be positive, got {self.nnls_maxiter}.")
        if float(self.nnls_tol) <= 0.0:
            raise ValueError(f"nnls_tol must be > 0, got {self.nnls_tol}.")
        if int(self.nnls_lipschitz_iters) <= 0:
            raise ValueError(f"nnls_lipschitz_iters must be positive, got {self.nnls_lipschitz_iters}.")

    @property
    def canonical_backend(self) -> Literal["matrix", "operator"]:
        """
        Standardizes the inversion backend name to lowercase.
        Returns 'matrix' or 'operator'.
        """
        backend = str(self.inversion_backend).strip().lower()
        if backend == "matrix":
            return "matrix"
        if backend == "operator":
            return "operator"
        
        # This should theoretically not be reached due to __post_init__ validation
        raise ValueError(f"Unknown inversion_backend: '{self.inversion_backend}'.")


@dataclass(frozen=True)
class PixelizedSourceConfig:
    """
    Top-level configuration for pixelized-source modeling.
    
    This class aggregates all sub-configurations required to define a pixelized 
    source model in a lensing simulation or inference pipeline. It serves as 
    the primary interface for configuring pixelized source reconstructions.

    Attributes:
        grid (GridConfig): Geometry configuration for the source plane pixels.
            Can be either IrregularGridConfig or RectangularGridConfig.
            Default factory creates IrregularGridConfig.
        mapping (MappingConfig): Interpolation settings for mapping source 
            surface brightness to the image plane. Default factory creates MappingConfig.
        regularization (RegularizationConfig): Constraints and smoothing operators 
            applied to the source. Default factory creates RegularizationConfig.
        solver (SolverConfig): Parameters for the linear system inversion and 
            likelihood calculation. Default factory creates SolverConfig.
    """

    grid: GridConfig = field(default_factory=IrregularGridConfig)
    mapping: MappingConfig = field(default_factory=MappingConfig)
    regularization: RegularizationConfig = field(default_factory=RegularizationConfig)
    solver: SolverConfig = field(default_factory=SolverConfig)

    def __post_init__(self) -> None:
        """
        Cross-validate sub-configurations for consistency.
        Ensures that the grid type and regularization scheme are compatible.
        """
        # Ensure rectangular grid is paired with a rectangular regularization scheme
        if isinstance(self.grid, RectangularGridConfig) and not self.regularization.is_rectangular_scheme:
            raise ValueError(
                "RectangularGridConfig requires a rectangular regularization scheme "
                "('rectangular_zero', 'rectangular_first', or 'rectangular_second')."
            )
        
        # Ensure irregular grid is paired with an irregular regularization scheme
        if isinstance(self.grid, IrregularGridConfig) and not self.regularization.is_irregular_scheme:
            raise ValueError(
                "IrregularGridConfig requires an irregular regularization scheme "
                "('irregular_gp_*' or 'irregular_knn_*')."
            )

    @property
    def is_rectangular(self) -> bool:
        """
        Returns True if the source plane is discretized using a rectangular grid.
        """
        return isinstance(self.grid, RectangularGridConfig)

    @property
    def source_grid_type(self) -> str:
        """
        Returns a string identifier of the source grid type.
        Used for downstream logic to dispatch to appropriate implementation classes.
        """
        return "rectangular_bilinear" if self.is_rectangular else "irregular"
