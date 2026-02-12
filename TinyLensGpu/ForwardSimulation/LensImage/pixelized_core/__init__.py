"""Core building blocks for pixelized-source simulation pipeline."""

from .artifacts import (
    GridArtifacts,
    MappingArtifacts,
    OperatorCacheKey,
    RegularizationArtifacts,
)
from .grid_strategies import (
    BaseGridStrategy,
    IrregularGridStrategy,
    RectangularGridStrategy,
)
from .mapping_strategies import (
    BaseMappingStrategy,
    KnnKernelMappingStrategy,
    RectBilinearMappingStrategy,
    build_mapping_artifacts,
)
from .regularization_strategies import (
    BaseRegularizationStrategy,
    DenseGpRegularizationStrategy,
    SparseKnnRegularizationStrategy,
    SparseRectangularRegularizationStrategy,
    select_regularization_strategy,
)
from .inversion_assembler import InversionAssembler

__all__ = [
    "GridArtifacts",
    "MappingArtifacts",
    "RegularizationArtifacts",
    "OperatorCacheKey",
    "BaseGridStrategy",
    "IrregularGridStrategy",
    "RectangularGridStrategy",
    "BaseMappingStrategy",
    "KnnKernelMappingStrategy",
    "RectBilinearMappingStrategy",
    "build_mapping_artifacts",
    "BaseRegularizationStrategy",
    "DenseGpRegularizationStrategy",
    "SparseKnnRegularizationStrategy",
    "SparseRectangularRegularizationStrategy",
    "select_regularization_strategy",
    "InversionAssembler",
]

