"""Assembler for matrix/operator inversion objects."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

import jax.numpy as jnp

from TinyLensGpu.PhysicalModel.LensImage.Pixelized.config import SolverConfig
from TinyLensGpu.utils.inversion import (
    LinearInversion,
    NNLSInversion,
    OperatorInversion,
    OperatorNNLSInversion,
)
from TinyLensGpu.utils.lensing import sparse_regularization_dense_from

from .artifacts import MappingArtifacts, RegularizationArtifacts


InversionType = Union[LinearInversion, OperatorInversion, NNLSInversion, OperatorNNLSInversion]


@dataclass(frozen=True)
class InversionAssembler:
    """
    Represent the `InversionAssembler` component in the TinyLensGpu pipeline.
    
    Notes
    -----
    Instances of this class participate in TinyLensGpu forward modeling and/or
    inference workflows. Keep parameter semantics consistent with neighboring
    modules to ensure predictable numerical behavior.
    """

    psf_fft: jnp.ndarray
    image_shape: tuple[int, int]
    psf_shape: tuple[int, int]
    unmasked_indices: tuple[jnp.ndarray, jnp.ndarray]

    def _regularization_dense_from(self, reg: RegularizationArtifacts) -> jnp.ndarray:
        """
        Internal helper to regularization dense from.
        
        Parameters
        ----------
        reg : Any
            Input argument used by this routine. Shapes/units follow the surrounding
            simulation or inference convention in the calling context.
        
        Returns
        -------
        value : Any
            Computed output produced by this routine. For array outputs, shape follows
            the input mesh/matrix conventions used by the corresponding pipeline stage.
        
        Raises
        ------
        RuntimeError
            Raised when input validation fails or required runtime state is missing.
        
        """
        if reg.is_sparse:
            if reg.sparse_rows is None or reg.sparse_cols is None or reg.sparse_values is None or reg.sparse_n_source is None:
                raise RuntimeError("Sparse regularization entries are incomplete.")
            return sparse_regularization_dense_from(
                rows=reg.sparse_rows,
                cols=reg.sparse_cols,
                values=reg.sparse_values,
                n_source=int(reg.sparse_n_source),
            )
        if reg.dense_matrix is None:
            raise RuntimeError("Dense regularization matrix is missing.")
        return reg.dense_matrix

    def _regularization_for_operator(self, reg: RegularizationArtifacts, n_source: int) -> jnp.ndarray:
        """
        Internal helper to regularization for operator.
        
        Parameters
        ----------
        reg : Any
            Input argument used by this routine. Shapes/units follow the surrounding
            simulation or inference convention in the calling context.
        n_source : Any
            Input argument used by this routine. Shapes/units follow the surrounding
            simulation or inference convention in the calling context.
        
        Returns
        -------
        value : Any
            Computed output produced by this routine. For array outputs, shape follows
            the input mesh/matrix conventions used by the corresponding pipeline stage.
        
        Raises
        ------
        RuntimeError
            Raised when input validation fails or required runtime state is missing.
        
        """
        if reg.is_sparse:
            return jnp.zeros((int(n_source),), dtype=jnp.float32)
        if reg.dense_matrix is None:
            raise RuntimeError("Dense regularization matrix is missing.")
        return reg.dense_matrix

    def build(
        self,
        *,
        data_vector: jnp.ndarray,
        noise_variance: jnp.ndarray,
        mapping: MappingArtifacts,
        regularization: RegularizationArtifacts,
        solver_config: SolverConfig,
        lens_basis_matrix: Optional[jnp.ndarray] = None,
    ) -> InversionType:
        """
        Compute build.
        
        Parameters
        ----------
        data_vector : Any
            Input argument used by this routine. Shapes/units follow the surrounding
            simulation or inference convention in the calling context.
        noise_variance : Any
            Input argument used by this routine. Shapes/units follow the surrounding
            simulation or inference convention in the calling context.
        mapping : Any
            Input argument used by this routine. Shapes/units follow the surrounding
            simulation or inference convention in the calling context.
        regularization : Any
            Input argument used by this routine. Shapes/units follow the surrounding
            simulation or inference convention in the calling context.
        solver_config : Any
            Input argument used by this routine. Shapes/units follow the surrounding
            simulation or inference convention in the calling context.
        lens_basis_matrix : Any
            Input argument used by this routine. Shapes/units follow the surrounding
            simulation or inference convention in the calling context.
        
        Returns
        -------
        value : Any
            Computed output produced by this routine. For array outputs, shape follows
            the input mesh/matrix conventions used by the corresponding pipeline stage.
        
        Raises
        ------
        RuntimeError
            Raised when input validation fails or required runtime state is missing.
        
        """
        backend = solver_config.canonical_backend
        d = jnp.asarray(data_vector)
        noise_var = jnp.asarray(noise_variance)

        if backend == "operator":
            if mapping.operator_weights is None or mapping.operator_indices is None:
                raise RuntimeError("Operator backend requires operator mapping weights/indices.")

            n_source = int(regularization.n_source)
            reg_matrix = self._regularization_for_operator(regularization, n_source=n_source)

            kwargs = dict(
                d=d,
                noise_var=noise_var,
                H=reg_matrix,
                weights=mapping.operator_weights,
                indices=mapping.operator_indices,
                psf_fft=self.psf_fft,
                image_shape=self.image_shape,
                psf_shape=self.psf_shape,
                unmasked_indices=self.unmasked_indices,
                slq_seed=solver_config.slq_seed,
                slq_probes=solver_config.slq_probes,
                slq_steps=solver_config.slq_steps,
                reg_operator_mode=regularization.mode,
                lens_basis=lens_basis_matrix,
                lens_light_ridge=solver_config.lens_light_ridge,
                H_sparse_rows=regularization.sparse_rows,
                H_sparse_cols=regularization.sparse_cols,
                H_sparse_values=regularization.sparse_values,
                H_sparse_n_source=regularization.sparse_n_source,
            )
            if solver_config.nonnegative:
                return OperatorNNLSInversion(
                    maxiter=solver_config.nnls_maxiter,
                    tol=solver_config.nnls_tol,
                    lipschitz_iters=solver_config.nnls_lipschitz_iters,
                    fista_seed=solver_config.slq_seed,
                    **kwargs,
                )
            return OperatorInversion(
                cg_tol=solver_config.cg_tol,
                cg_maxiter=solver_config.cg_maxiter,
                **kwargs,
            )

        if mapping.dense_matrix is None:
            raise RuntimeError("Matrix backend requires dense mapping matrix.")

        reg_matrix_src = self._regularization_dense_from(regularization)
        blurred_lens_map_matrix = mapping.dense_matrix

        if lens_basis_matrix is None:
            if solver_config.nonnegative:
                return NNLSInversion(d=d, F=blurred_lens_map_matrix, noise_cov=noise_var, H=reg_matrix_src)
            return LinearInversion(d=d, F=blurred_lens_map_matrix, noise_cov=noise_var, H=reg_matrix_src)

        f_total = jnp.concatenate([blurred_lens_map_matrix, lens_basis_matrix], axis=1)
        n_src = int(reg_matrix_src.shape[0])
        n_lens = int(lens_basis_matrix.shape[1])
        ridge = jnp.array(max(float(solver_config.lens_light_ridge), 0.0), dtype=jnp.float32)
        h_lens = ridge * jnp.eye(n_lens, dtype=jnp.float32)
        h_total = jnp.block(
            [
                [reg_matrix_src, jnp.zeros((n_src, n_lens), dtype=jnp.float32)],
                [jnp.zeros((n_lens, n_src), dtype=jnp.float32), h_lens],
            ]
        )
        if solver_config.nonnegative:
            return NNLSInversion(d=d, F=f_total, noise_cov=noise_var, H=h_total)
        return LinearInversion(d=d, F=f_total, noise_cov=noise_var, H=h_total)
