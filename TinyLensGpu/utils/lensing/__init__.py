"""Lensing operation utilities."""

from .mapping import (
    dense_mapping_from_weights_indices,
    lens_mapping_operator_bilinear_rectangular_from,
    build_source_grid,
    build_lens_mapping_matrix,
    infer_source_bbox,
)
from .psf import build_psf_matrix_dense, build_psf_matrix_sparse, apply_psf_to_mapping_matrix
from .point_source_solver import (
    solve_lens_equation_optimization_core,
    solve_lens_equation_mesh_refine_core,
    solve_lens_equation_optimization,
    solve_lens_equation_mesh_refine,
    post_process_images,
    select_unique_images_fixed,
    build_permutation_indices,
    min_assignment_chi2,
)

__all__ = [
    'dense_mapping_from_weights_indices',
    'lens_mapping_operator_bilinear_rectangular_from',
    'build_source_grid',
    'build_lens_mapping_matrix',
    'infer_source_bbox',
    'build_psf_matrix_dense',
    'build_psf_matrix_sparse',
    'apply_psf_to_mapping_matrix',
    'solve_lens_equation_optimization_core',
    'solve_lens_equation_mesh_refine_core',
    'solve_lens_equation_optimization',
    'solve_lens_equation_mesh_refine',
    'post_process_images',
    'select_unique_images_fixed',
    'build_permutation_indices',
    'min_assignment_chi2',
]
