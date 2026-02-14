"""Forward model for pixelized source gravitational lensing."""

from __future__ import annotations

import functools
from typing import Optional, Union

import jax.numpy as jnp
import numpy as np
from jax import Array, jit

from .config import SimulatorConfig
from TinyLensGpu.PhysicalModel.LensImage.Pixelized.config import (
    IrregularGridConfig,
    RectangularGridConfig,
)
from TinyLensGpu.PhysicalModel.LensImage.composite import PhysicalModel
from TinyLensGpu.utils.inversion import LinearInversion, NNLSInversion, OperatorInversion, OperatorNNLSInversion
from TinyLensGpu.utils.lensing import apply_psf_to_mapping_matrix

from .pixelized_core import (
    DenseGpRegularizationStrategy,
    GridArtifacts,
    InversionAssembler,
    IrregularGridStrategy,
    KnnKernelMappingStrategy,
    MappingArtifacts,
    RectBilinearMappingStrategy,
    RectangularGridStrategy,
    RegularizationArtifacts,
    SparseKnnRegularizationStrategy,
    SparseRectangularRegularizationStrategy,
)

class PixelizedLensSimulator:
    def __init__(
        self,
        phys_model: PhysicalModel,
        sim_config: SimulatorConfig,
        lensed_source_image: Optional[np.ndarray] = None,
    ) -> None:
        self.sim_config = sim_config
        self.dpix = float(sim_config.dpix)
        self.npix = int(sim_config.npix)
        self.phys_model = phys_model
        self.pix_src_model = self.phys_model.get_pixelized_source_model()
        self.psf_kernel = jnp.asarray(sim_config.psf_kernel)
        if self.psf_kernel.ndim != 2:
            raise ValueError(f"sim_config.psf_kernel must be 2D, got shape {self.psf_kernel.shape}.")

        self.mask = jnp.asarray(sim_config.mask, dtype=bool)
        expected_shape = (self.npix, self.npix)
        if self.mask.shape != expected_shape:
            raise ValueError(
                f"sim_config.mask shape mismatch: expected {expected_shape}, got {self.mask.shape}."
            )

        xgrid_2d = jnp.asarray(sim_config.xgrid)
        ygrid_2d = jnp.asarray(sim_config.ygrid)
        if xgrid_2d.shape != expected_shape or ygrid_2d.shape != expected_shape:
            raise ValueError(
                "sim_config grid shape mismatch: "
                f"expected {expected_shape}, got xgrid={xgrid_2d.shape}, ygrid={ygrid_2d.shape}."
            )

        if lensed_source_image is None:
            lensed_source_image = np.ones(expected_shape, dtype=np.float32)
        else:
            if np.asarray(lensed_source_image).shape != expected_shape:
                raise ValueError(
                    "lensed_source_image shape mismatch: "
                    f"expected {expected_shape}, got {np.asarray(lensed_source_image).shape}."
                )
        self.lensed_source_image = jnp.asarray(lensed_source_image)

        self.xgrid_unmask = jnp.asarray(xgrid_2d[~self.mask], dtype=jnp.float32)
        self.ygrid_unmask = jnp.asarray(ygrid_2d[~self.mask], dtype=jnp.float32)

        y_indices, x_indices = np.where(~np.asarray(self.mask))
        self.unmasked_indices = (jnp.asarray(y_indices), jnp.asarray(x_indices))
        self.image_shape = (self.npix, self.npix)

        psf_h, psf_w = self.psf_kernel.shape
        self.psf_shape = (int(psf_h), int(psf_w))
        self.fft_shape = (self.npix + int(psf_h) - 1, self.npix + int(psf_w) - 1)
        self.psf_fft = jnp.fft.rfft2(self.psf_kernel, s=self.fft_shape)

        self._cached_operator_weights = None
        self._cached_operator_indices = None
        self._cached_operator_signature = None

        self._grid_artifacts: Optional[GridArtifacts] = None
        self._mapping_strategy = self._build_mapping_strategy()
        self._regularization_strategy = self._build_regularization_strategy()
        self._inversion_assembler = InversionAssembler(
            psf_fft=self.psf_fft,
            image_shape=self.image_shape,
            psf_shape=self.psf_shape,
            unmasked_indices=self.unmasked_indices,
        )

        self._build_grid_artifacts(self.lensed_source_image, self.mask)

    def _build_mapping_strategy(self):
        if self.pix_src_model.is_rectangular_grid:
            return RectBilinearMappingStrategy()
        return KnnKernelMappingStrategy(config=self.pix_src_model.mapping)

    def _build_regularization_strategy(self):
        mode = self.pix_src_model.regularization.resolved_mode(self.pix_src_model.grid)
        if mode == "dense_gp":
            return DenseGpRegularizationStrategy(config=self.pix_src_model.regularization)
        if mode == "sparse_knn":
            return SparseKnnRegularizationStrategy(config=self.pix_src_model.regularization)
        if mode == "sparse_rectangular":
            return SparseRectangularRegularizationStrategy(config=self.pix_src_model.regularization)
        raise ValueError(f"Unknown regularization mode: '{mode}'.")

    @functools.partial(jit, static_argnums=(0,))
    def ray_trace(self, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        beta_x, beta_y = self.phys_model.deflection(x=x, y=y)
        return jnp.stack([beta_x, beta_y], axis=1)

    @property
    def source_mesh(self) -> jnp.ndarray:
        if self._grid_artifacts is None:
            raise RuntimeError("Grid artifacts are not initialized.")
        return self._grid_artifacts.source_mesh

    @property
    def source_mesh_beta(self) -> jnp.ndarray:
        if self._grid_artifacts is None:
            raise RuntimeError("Grid artifacts are not initialized.")
        return self._grid_artifacts.source_mesh_beta

    @property
    def data_mesh_beta(self) -> jnp.ndarray:
        if self._grid_artifacts is None:
            raise RuntimeError("Grid artifacts are not initialized.")
        return self._grid_artifacts.data_mesh_beta

    @property
    def source_grid_shape(self):
        if self._grid_artifacts is None:
            return None
        return self._grid_artifacts.source_grid_shape

    @property
    def source_grid_bounds(self):
        if self._grid_artifacts is None:
            return None
        return self._grid_artifacts.source_grid_bounds

    def _build_grid_artifacts(self, lensed_source_image: Array | np.ndarray, mask: Array | np.ndarray) -> None:
        data_mesh_beta = self.ray_trace(self.xgrid_unmask, self.ygrid_unmask)
        if isinstance(self.pix_src_model.grid, RectangularGridConfig):
            strategy = RectangularGridStrategy(config=self.pix_src_model.grid)
        elif isinstance(self.pix_src_model.grid, IrregularGridConfig):
            strategy = IrregularGridStrategy(config=self.pix_src_model.grid)
        else:
            raise TypeError("Unknown grid config type.")

        self._grid_artifacts = strategy.build(
            lensed_source_image=np.asarray(lensed_source_image),
            mask=np.asarray(mask),
            dpix=self.dpix,
            data_mesh_beta=data_mesh_beta,
            ray_trace=self.ray_trace,
        )

    def clear_operator_cache(self) -> None:
        self._cached_operator_weights = None
        self._cached_operator_indices = None
        self._cached_operator_signature = None

    def _build_mapping_artifacts(self, *, backend: str, cache_policy: str) -> MappingArtifacts:
        if self._grid_artifacts is None:
            raise RuntimeError("Grid artifacts are not initialized.")

        need_dense = backend == "matrix"
        need_operator = backend == "operator"

        dense_matrix = None
        op_weights = None
        op_indices = None

        if need_dense:
            dense_matrix = self._mapping_strategy.build_dense(self._grid_artifacts)

        if need_operator:
            policy = str(cache_policy).strip().lower()
            if policy == "unsafe_static" and self._cached_operator_weights is not None and self._cached_operator_indices is not None:
                op_weights, op_indices = self._cached_operator_weights, self._cached_operator_indices
            else:
                key = self._mapping_strategy.operator_cache_key(self._grid_artifacts)
                if (
                    policy == "safe"
                    and self._cached_operator_weights is not None
                    and self._cached_operator_indices is not None
                    and self._cached_operator_signature == key.signature
                ):
                    op_weights, op_indices = self._cached_operator_weights, self._cached_operator_indices
                else:
                    op_weights, op_indices = self._mapping_strategy.build_operator(self._grid_artifacts)
                    if policy in {"safe", "unsafe_static"}:
                        self._cached_operator_weights = op_weights
                        self._cached_operator_indices = op_indices
                        self._cached_operator_signature = "unsafe_static" if policy == "unsafe_static" else key.signature

        return MappingArtifacts(dense_matrix=dense_matrix, operator_weights=op_weights, operator_indices=op_indices)

    def _build_regularization_artifacts(self, reg_scale: float, reg_coefficient: float) -> RegularizationArtifacts:
        if self._grid_artifacts is None:
            raise RuntimeError("Grid artifacts are not initialized.")
        return self._regularization_strategy.build(
            grid=self._grid_artifacts,
            reg_scale=float(reg_scale),
            reg_coefficient=float(reg_coefficient),
        )

    def build_lens_mapping_matrix(self) -> jnp.ndarray:
        mapping = self._build_mapping_artifacts(backend="matrix", cache_policy="safe")
        if mapping.dense_matrix is None:
            raise RuntimeError("Dense mapping matrix was not produced.")
        return mapping.dense_matrix

    def build_lens_mapping_operator(
        self,
        *,
        use_cache: bool = True,
        cache_policy: str = "safe",
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        policy = cache_policy if use_cache else "off"
        mapping = self._build_mapping_artifacts(backend="operator", cache_policy=policy)
        if mapping.operator_weights is None or mapping.operator_indices is None:
            raise RuntimeError("Operator mapping entries were not produced.")
        return mapping.operator_weights, mapping.operator_indices

    def build_blurred_lens_mapping_matrix(self, method: str = "fft") -> jnp.ndarray:
        return apply_psf_to_mapping_matrix(
            mapping_matrix=self.build_lens_mapping_matrix(),
            psf_kernel=self.psf_kernel,
            image_shape=self.image_shape,
            unmasked_indices=self.unmasked_indices,
            method=method,
            psf_fft=self.psf_fft if method == "fft" else None,
            psf_shape=self.psf_shape if method == "fft" else None,
        )

    def build_lens_light_basis_matrix(self, method: str = "fft") -> jnp.ndarray:
        n_lens_light = len(self.phys_model.lens_light)
        if n_lens_light == 0:
            return jnp.zeros((self.xgrid_unmask.shape[0], 0), dtype=jnp.float32)

        basis = []
        for light_model in self.phys_model.lens_light:
            basis.append(light_model.light(x=self.xgrid_unmask, y=self.ygrid_unmask))
        basis_unblurred = jnp.stack(basis, axis=1).astype(jnp.float32)

        return apply_psf_to_mapping_matrix(
            mapping_matrix=basis_unblurred,
            psf_kernel=self.psf_kernel,
            image_shape=self.image_shape,
            unmasked_indices=self.unmasked_indices,
            method=method,
            psf_fft=self.psf_fft if method == "fft" else None,
            psf_shape=self.psf_shape if method == "fft" else None,
        )

    def build_regularization_matrix(self, reg_scale: float, reg_coefficient: float) -> jnp.ndarray:
        reg = self._build_regularization_artifacts(reg_scale=reg_scale, reg_coefficient=reg_coefficient)
        if reg.dense_matrix is not None:
            return reg.dense_matrix
        from TinyLensGpu.utils.lensing import sparse_regularization_dense_from

        if reg.sparse_rows is None or reg.sparse_cols is None or reg.sparse_values is None or reg.sparse_n_source is None:
            raise RuntimeError("Sparse regularization artifacts are incomplete.")
        return sparse_regularization_dense_from(
            rows=reg.sparse_rows,
            cols=reg.sparse_cols,
            values=reg.sparse_values,
            n_source=reg.sparse_n_source,
        )

    def build_inverter(
        self,
        data_vector: Union[jnp.ndarray, np.ndarray],
        noise_variance: Union[jnp.ndarray, np.ndarray],
        reg_scale: float,
        reg_coefficient: float,
    ) -> Union[LinearInversion, OperatorInversion, NNLSInversion, OperatorNNLSInversion]:
        solver_cfg = self.pix_src_model.solver
        backend = solver_cfg.canonical_backend

        reg = self._build_regularization_artifacts(reg_scale=reg_scale, reg_coefficient=reg_coefficient)
        mapping = self._build_mapping_artifacts(backend=backend, cache_policy=solver_cfg.operator_cache_policy)

        if backend == "matrix" and mapping.dense_matrix is not None:
            mapping = MappingArtifacts(
                dense_matrix=apply_psf_to_mapping_matrix(
                    mapping_matrix=mapping.dense_matrix,
                    psf_kernel=self.psf_kernel,
                    image_shape=self.image_shape,
                    unmasked_indices=self.unmasked_indices,
                    method="fft",
                    psf_fft=self.psf_fft,
                    psf_shape=self.psf_shape,
                ),
                operator_weights=mapping.operator_weights,
                operator_indices=mapping.operator_indices,
            )

        lens_basis_matrix = None
        if solver_cfg.include_lens_light:
            lens_basis_matrix = self.build_lens_light_basis_matrix()

        return self._inversion_assembler.build(
            data_vector=jnp.asarray(data_vector),
            noise_variance=jnp.asarray(noise_variance),
            mapping=mapping,
            regularization=reg,
            solver_config=solver_cfg,
            lens_basis_matrix=lens_basis_matrix,
        )

    def reconstruct_source_and_lens_light(
        self,
        data_vector: Union[jnp.ndarray, np.ndarray],
        noise_variance: Union[jnp.ndarray, np.ndarray],
        reg_scale: float,
        reg_coefficient: float,
        return_2d: bool = False,
    ):
        if not self.pix_src_model.solver.include_lens_light:
            raise ValueError(
                "SolverConfig.include_lens_light must be True for reconstruct_source_and_lens_light(). "
                "Please configure SolverConfig(include_lens_light=True) when creating the PixelizedSourceModel."
            )

        inverter = self.build_inverter(
            data_vector=data_vector,
            noise_variance=noise_variance,
            reg_scale=reg_scale,
            reg_coefficient=reg_coefficient,
        )

        x_total = inverter.solve()
        n_src = int(self.source_mesh_beta.shape[0])
        source_intensities = x_total[:n_src]
        lens_light_intensities = x_total[n_src:]
        model_data = inverter.model_predict(x_total)

        if return_2d:
            model_image = jnp.zeros((self.npix, self.npix))
            model_image = model_image.at[~self.mask].set(model_data)
        else:
            model_image = model_data

        return source_intensities, lens_light_intensities, self.source_mesh_beta, model_image, inverter

    def reconstruct_source(
        self,
        data_vector: Union[jnp.ndarray, np.ndarray],
        noise_variance: Union[jnp.ndarray, np.ndarray],
        reg_scale: float,
        reg_coefficient: float,
        return_2d: bool = False,
    ) -> Tuple[
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
        Union[LinearInversion, OperatorInversion, NNLSInversion, OperatorNNLSInversion],
    ]:
        inverter = self.build_inverter(
            data_vector=data_vector,
            noise_variance=noise_variance,
            reg_scale=reg_scale,
            reg_coefficient=reg_coefficient,
        )
        source_intensities = inverter.solve()
        model_data = inverter.model_predict(source_intensities)

        if return_2d:
            model_image = jnp.zeros((self.npix, self.npix))
            model_image = model_image.at[~self.mask].set(model_data)
        else:
            model_image = model_data

        return source_intensities, self.source_mesh_beta, model_image, inverter

    def __repr__(self) -> str:
        if isinstance(self.pix_src_model.grid, IrregularGridConfig):
            n_source_points = int(self.pix_src_model.grid.n_source_points)
        elif isinstance(self.pix_src_model.grid, RectangularGridConfig):
            n_source_points = int(self.pix_src_model.grid.nx * self.pix_src_model.grid.ny)
        else:
            n_source_points = 0

        return (
            "PixelizedLensSimulator("
            f"npix={self.npix}, "
            f"n_source_points={n_source_points}, "
            f"n_mass={len(self.phys_model.lens_mass)})"
        )
