"""Configuration models for pixelized-source reconstruction."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional, Tuple, Union


REG_KERNEL_TYPES = {"exp", "gauss", "matern32", "matern52"}
MESH_METHODS = {"random", "sobol"}
INTERP_KERNELS = {"wendland_c2", "wendland_c4", "wendland_c6"}
REG_SCHEMES = {"zero", "gradient", "curvature"}
REG_MODES = {"auto", "dense_gp", "sparse_knn", "sparse_rectangular"}
SOLVER_BACKENDS = {"matrix", "operator"}


@dataclass(frozen=True)
class IrregularGridConfig:
    """Configuration for irregular source-plane mesh sampling."""

    n_source_points: int = 1500
    mesh_alpha: float = 0.0
    mesh_blur_sigma: float = 0.0
    mesh_method: Literal["random", "sobol"] = "random"
    mesh_seed: int = 42

    def __post_init__(self) -> None:
        if int(self.n_source_points) <= 0:
            raise ValueError("n_source_points must be positive.")
        if float(self.mesh_blur_sigma) < 0.0:
            raise ValueError("mesh_blur_sigma must be >= 0.")
        if str(self.mesh_method).strip().lower() not in MESH_METHODS:
            raise ValueError(
                f"Unknown mesh_method: '{self.mesh_method}'. Must be one of {MESH_METHODS}."
            )


@dataclass(frozen=True)
class RectangularGridConfig:
    """Configuration for rectangular source-plane grids."""

    nx: int = 64
    ny: int = 64
    margin_frac: float = 0.10
    bounds: Optional[Tuple[float, float, float, float]] = None

    def __post_init__(self) -> None:
        if int(self.nx) <= 0 or int(self.ny) <= 0:
            raise ValueError(f"Rectangular grid shape must be positive, got nx={self.nx}, ny={self.ny}.")
        if float(self.margin_frac) < 0.0:
            raise ValueError("margin_frac must be >= 0.")
        if self.bounds is not None:
            if len(self.bounds) != 4:
                raise ValueError("bounds must be a 4-tuple: (x_min, x_max, y_min, y_max).")
            x_min, x_max, y_min, y_max = [float(v) for v in self.bounds]
            if x_max <= x_min or y_max <= y_min:
                raise ValueError(
                    "bounds must satisfy x_max>x_min and y_max>y_min, got "
                    f"({x_min}, {x_max}, {y_min}, {y_max})."
                )


GridConfig = Union[IrregularGridConfig, RectangularGridConfig]


@dataclass(frozen=True)
class MappingConfig:
    """Configuration for source-to-image interpolation mapping."""

    k_neighbors: int = 5
    interp_kernel: Literal["wendland_c2", "wendland_c4", "wendland_c6"] = "wendland_c4"
    radius_scale: float = 1.5

    def __post_init__(self) -> None:
        if int(self.k_neighbors) <= 0:
            raise ValueError("k_neighbors must be positive.")
        if str(self.interp_kernel).strip().lower() not in INTERP_KERNELS:
            raise ValueError(
                f"Unknown interp_kernel: '{self.interp_kernel}'. Must be one of {INTERP_KERNELS}."
            )
        if float(self.radius_scale) <= 0.0:
            raise ValueError("radius_scale must be > 0.")


@dataclass(frozen=True)
class RegularizationConfig:
    """Configuration for source regularization operators."""

    mode: Literal["auto", "dense_gp", "sparse_knn", "sparse_rectangular"] = "auto"
    gp_kernel: Literal["exp", "gauss", "matern32", "matern52"] = "exp"
    sparse_k_neighbors: int = 16
    rect_scheme: Literal["zero", "gradient", "curvature"] = "gradient"

    def __post_init__(self) -> None:
        if str(self.mode).strip().lower() not in REG_MODES:
            raise ValueError(f"Unknown regularization mode: '{self.mode}'. Must be one of {REG_MODES}.")
        if str(self.gp_kernel).strip().lower() not in REG_KERNEL_TYPES:
            raise ValueError(f"Unknown gp_kernel: '{self.gp_kernel}'. Must be one of {REG_KERNEL_TYPES}.")
        if int(self.sparse_k_neighbors) <= 0:
            raise ValueError("sparse_k_neighbors must be positive.")
        if str(self.rect_scheme).strip().lower() not in REG_SCHEMES:
            raise ValueError(f"Unknown rect_scheme: '{self.rect_scheme}'. Must be one of {REG_SCHEMES}.")

    def resolved_mode(self, grid: GridConfig) -> Literal["dense_gp", "sparse_knn", "sparse_rectangular"]:
        """Resolve the runtime regularization mode from config + grid type."""
        mode = str(self.mode).strip().lower()
        if mode != "auto":
            return mode  # type: ignore[return-value]
        if isinstance(grid, RectangularGridConfig):
            return "sparse_rectangular"
        return "dense_gp"


@dataclass(frozen=True)
class SolverConfig:
    """Configuration for linear inversion backend behavior."""

    inversion_backend: Literal["matrix", "operator"] = "matrix"
    include_lens_light: bool = False
    nonnegative: bool = False
    lens_light_ridge: float = 1e-8

    cg_tol: float = 1e-4
    cg_maxiter: int = 120
    slq_seed: int = 0
    slq_probes: int = 32
    slq_steps: int = 60
    evidence_mode: str = "accurate"
    operator_cache_policy: str = "safe"

    nnls_maxiter: int = 600
    nnls_tol: float = 1e-6
    nnls_lipschitz_iters: int = 12

    def __post_init__(self) -> None:
        if str(self.inversion_backend).strip().lower() not in SOLVER_BACKENDS:
            raise ValueError(
                f"Unknown inversion_backend: '{self.inversion_backend}'. Must be one of {SOLVER_BACKENDS}."
            )
        if float(self.lens_light_ridge) < 0.0:
            raise ValueError("lens_light_ridge must be >= 0.")
        if float(self.cg_tol) <= 0.0:
            raise ValueError("cg_tol must be > 0.")
        if int(self.cg_maxiter) <= 0:
            raise ValueError("cg_maxiter must be positive.")
        if int(self.slq_probes) <= 0 or int(self.slq_steps) <= 0:
            raise ValueError("slq_probes and slq_steps must be positive.")
        if int(self.nnls_maxiter) <= 0:
            raise ValueError("nnls_maxiter must be positive.")
        if float(self.nnls_tol) <= 0.0:
            raise ValueError("nnls_tol must be > 0.")
        if int(self.nnls_lipschitz_iters) <= 0:
            raise ValueError("nnls_lipschitz_iters must be positive.")

    @property
    def canonical_backend(self) -> Literal["matrix", "operator"]:
        backend = str(self.inversion_backend).strip().lower()
        if backend == "matrix":
            return "matrix"
        if backend == "operator":
            return "operator"
        raise ValueError(f"Unknown inversion_backend: '{self.inversion_backend}'.")


@dataclass(frozen=True)
class PixelizedSourceConfig:
    """Top-level configuration for pixelized-source modeling."""

    grid: GridConfig = field(default_factory=IrregularGridConfig)
    mapping: MappingConfig = field(default_factory=MappingConfig)
    regularization: RegularizationConfig = field(default_factory=RegularizationConfig)
    solver: SolverConfig = field(default_factory=SolverConfig)

    def __post_init__(self) -> None:
        mode = self.regularization.resolved_mode(self.grid)
        if isinstance(self.grid, RectangularGridConfig) and mode != "sparse_rectangular":
            raise ValueError(
                "RectangularGridConfig requires regularization mode 'sparse_rectangular' or 'auto'."
            )
        if isinstance(self.grid, IrregularGridConfig) and mode == "sparse_rectangular":
            raise ValueError("IrregularGridConfig cannot use regularization mode 'sparse_rectangular'.")

    @property
    def is_rectangular(self) -> bool:
        return isinstance(self.grid, RectangularGridConfig)

    @property
    def source_grid_type(self) -> str:
        return "rectangular_bilinear" if self.is_rectangular else "irregular"
