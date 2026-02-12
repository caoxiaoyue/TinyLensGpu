"""Lensing operation utilities."""

from .mapping import (
    lens_mapping_matrix_from,
    lens_mapping_operator_bilinear_rectangular_from,
    lens_mapping_matrix_bilinear_rectangular_from,
)
from .psf import build_psf_matrix_dense, build_psf_matrix_sparse, apply_psf_to_mapping_matrix
from .regularization import (
    exp_cov_matrix_from,
    gauss_cov_matrix_from,
    matern32_cov_matrix_from,
    matern52_cov_matrix_from,
    regularization_matrix_gp_from,
    regularization_sparse_knn_from,
    regularization_sparse_rectangular_from,
    sparse_regularization_dense_from,
)
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
    'lens_mapping_matrix_from',
    'lens_mapping_operator_bilinear_rectangular_from',
    'lens_mapping_matrix_bilinear_rectangular_from',
    'build_psf_matrix_dense',
    'build_psf_matrix_sparse',
    'apply_psf_to_mapping_matrix',
    'exp_cov_matrix_from',
    'gauss_cov_matrix_from',
    'matern32_cov_matrix_from',
    'matern52_cov_matrix_from',
    'regularization_matrix_gp_from',
    'regularization_sparse_knn_from',
    'regularization_sparse_rectangular_from',
    'sparse_regularization_dense_from',
    'solve_lens_equation_optimization_core',
    'solve_lens_equation_mesh_refine_core',
    'solve_lens_equation_optimization',
    'solve_lens_equation_mesh_refine',
    'post_process_images',
    'select_unique_images_fixed',
    'build_permutation_indices',
    'min_assignment_chi2',
]
