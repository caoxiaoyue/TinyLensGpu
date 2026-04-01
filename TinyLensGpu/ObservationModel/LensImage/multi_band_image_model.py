# pyright: reportMissingImports=false

"""
Multi-band probability model scaffolding for lensing image fitting.

This module provides a minimal observation-layer wrapper for assembling
per-band image metadata and physical models with constructor-time validation.
"""

from dataclasses import dataclass, replace
from typing import Any, Dict, Optional, Sequence, Tuple, Union, cast

import caskade as ck
import jax.numpy as jnp
import numpy as np
from jax import Array

from TinyLensGpu.PhysicalModel.LensImage.composite import PhysicalModel
from TinyLensGpu.ForwardSimulation.LensImage.config import make_grid_2d_transformed
from TinyLensGpu.ObservationModel.LensImage.parametric_image_model import ImageProbModel
from TinyLensGpu.Inference.param_u import ParamU


@dataclass(frozen=True)
class BandObservationGeometry:
    """
    Per-band observation geometry in the shared sky frame.

    Parameters
    ----------
    shift_x : float or ParamU, optional
        Apparent image-plane offset along +x in arcsec relative to the
        reference band.
    shift_y : float or ParamU, optional
        Apparent image-plane offset along +y in arcsec relative to the
        reference band.
    rotation : float or ParamU, optional
        Apparent image-plane rotation in degrees with positive values
        corresponding to counterclockwise rotation.
    is_reference : bool, optional
        Whether this band defines the common sky-frame reference geometry.
    """

    shift_x: Union[float, ParamU] = 0.0
    shift_y: Union[float, ParamU] = 0.0
    rotation: Union[float, ParamU] = 0.0
    is_reference: bool = False


@dataclass(frozen=True)
class BandImageData:
    """
    Container for one observed imaging band.

    Parameters
    ----------
    name : str
        Band identifier used for deterministic ordering and lookup.
    image_data : Union[np.ndarray, Array]
        Observed image for this band.
    noise_map : Union[np.ndarray, Array]
        Per-pixel noise standard deviation map.
    psf_kernel : Union[np.ndarray, Array]
        Point spread function kernel for this band.
    dpix : float
        Pixel scale in arcsec/pixel.
    nsub : int
        Subsampling factor for ray-tracing.
    mask : Optional[Union[np.ndarray, Array]], optional
        Optional boolean mask where True indicates masked pixels.
    geometry : BandObservationGeometry | None, optional
        Optional per-band geometry metadata in the common sky frame.
    """

    name: str
    image_data: Union[np.ndarray, Array]
    noise_map: Union[np.ndarray, Array]
    psf_kernel: Union[np.ndarray, Array]
    dpix: float
    nsub: int
    mask: Optional[Union[np.ndarray, Array]] = None
    geometry: BandObservationGeometry | None = None


class _BandAlignmentModule(ck.Module):
    def __init__(
        self,
        name: str,
        *,
        shift_x: Union[float, ParamU],
        shift_y: Union[float, ParamU],
        rotation: Union[float, ParamU],
    ) -> None:
        super().__init__(name)
        self.shift_x = self._as_param(f"{name}_shift_x", shift_x)
        self.shift_y = self._as_param(f"{name}_shift_y", shift_y)
        self.rotation = self._as_param(f"{name}_rotation", rotation)

    @staticmethod
    def _as_param(param_name: str, value: Union[float, ParamU]) -> ParamU:
        if isinstance(value, ParamU):
            return value
        return ParamU(param_name, float(value))


class MultiBandImageProbModel(ck.Module):
    """
    Multi-band image probability model for joint fitting across observation bands.

    This class validates multi-band constructor inputs, manages deterministic band
    ordering, and evaluates the joint log-likelihood by summing per-band log-likelihoods.
    Per-band alignment transforms (shifts and rotation) are applied when bands have
    non-identity geometry relative to the reference sky frame.

    Parameters
    ----------
    bands : Sequence[BandImageData]
        Observed data containers for each band.
    phys_models : Sequence[PhysicalModel]
        Physical models aligned one-to-one with ``bands``.
    use_linear : bool
        Whether linear solving is enabled for per-band simulators.
    solver_type : str, optional
        Linear solver type, by default ``'nnls'``.

    Raises
    ------
    ValueError
        If ``bands`` is empty, band/model counts mismatch, or names are duplicated.
    """

    def __init__(
        self,
        bands: Sequence[BandImageData],
        phys_models: Sequence[PhysicalModel],
        *,
        use_linear: bool,
        solver_type: str = "nnls",
    ) -> None:
        super().__init__("multi_band_image_prob_model")

        if len(bands) == 0:
            raise ValueError("bands must be a non-empty sequence")

        if len(bands) != len(phys_models):
            raise ValueError("bands and phys_models must have the same length")

        normalized_bands = self._normalize_band_references(bands)

        band_names = [band.name for band in normalized_bands]
        for i, name in enumerate(band_names):
            if name == "":
                raise ValueError(
                    f"Band name at index {i} is empty. Each band must have a non-empty name."
                )
        if len(set(band_names)) != len(band_names):
            raise ValueError("band names must be unique")

        self._validate_band_geometry(normalized_bands)

        object.__setattr__(self, "bands", tuple(normalized_bands))
        object.__setattr__(self, "phys_models", tuple(phys_models))
        self.use_linear = use_linear
        self.solver_type = solver_type
        self.band_names = tuple(band_names)
        band_alignment_modules = self._build_band_alignment_modules()
        object.__setattr__(self, "_band_alignment_modules", band_alignment_modules)
        band_models = self._build_band_models()
        object.__setattr__(self, "band_models", band_models)
        object.__setattr__(
            self,
            "band_model_by_name",
            {band_name: band_model for band_name, band_model in zip(self.band_names, band_models)},
        )
        band_identity_geometry = tuple(
            self._is_identity_geometry(band.geometry) for band in self.bands
        )
        object.__setattr__(self, "_band_identity_geometry", band_identity_geometry)

        # Keep jnp imported and ready for subsequent multi-band likelihood work.
        self._num_bands = jnp.array(len(self.bands), dtype=jnp.int32)

    @staticmethod
    def _is_identity_geometry(geometry: BandObservationGeometry | None) -> bool:
        if geometry is None:
            return True
        if any(isinstance(value, ParamU) for value in (geometry.shift_x, geometry.shift_y, geometry.rotation)):
            return False
        return bool(
            np.isclose(geometry.shift_x, 0.0)
            and np.isclose(geometry.shift_y, 0.0)
            and np.isclose(geometry.rotation, 0.0)
        )

    @staticmethod
    def _normalize_band_references(bands: Sequence[BandImageData]) -> tuple[BandImageData, ...]:
        reference_indices = [
            idx
            for idx, band in enumerate(bands)
            if band.geometry is not None and band.geometry.is_reference
        ]
        if len(reference_indices) > 1:
            raise ValueError("at most one band may set geometry.is_reference=True")

        reference_idx = reference_indices[0] if len(reference_indices) == 1 else 0
        normalized_bands = []
        for idx, band in enumerate(bands):
            is_reference = idx == reference_idx
            if is_reference:
                normalized_geometry = BandObservationGeometry(
                    shift_x=0.0,
                    shift_y=0.0,
                    rotation=0.0,
                    is_reference=True,
                )
            elif band.geometry is None:
                normalized_geometry = BandObservationGeometry(is_reference=False)
            else:
                normalized_geometry = replace(band.geometry, is_reference=False)
            normalized_bands.append(replace(band, geometry=normalized_geometry))

        return tuple(normalized_bands)

    def _build_band_alignment_modules(self) -> tuple[_BandAlignmentModule, ...]:
        alignment_modules: list[_BandAlignmentModule] = []
        for idx, band in enumerate(self.bands):
            geometry = band.geometry if band.geometry is not None else BandObservationGeometry()
            module = _BandAlignmentModule(
                name=f"band_alignment_{idx}",
                shift_x=geometry.shift_x,
                shift_y=geometry.shift_y,
                rotation=geometry.rotation,
            )
            setattr(self, f"band_alignment_{idx}", module)
            alignment_modules.append(module)
        return tuple(alignment_modules)

    @staticmethod
    def _validate_band_geometry(bands: Sequence[BandImageData]) -> None:
        for band in bands:
            image_shape = np.shape(band.image_data)
            noise_shape = np.shape(band.noise_map)

            if len(image_shape) != 2:
                raise ValueError(f"band '{band.name}' image_data must be 2D, got shape {image_shape}")

            if image_shape[0] != image_shape[1]:
                raise ValueError(
                    f"band '{band.name}' image_data must be square; got shape {image_shape}"
                )

            if len(noise_shape) != 2:
                raise ValueError(f"band '{band.name}' noise_map must be 2D, got shape {noise_shape}")

            if noise_shape != image_shape:
                raise ValueError(
                    f"band '{band.name}' image_data shape {image_shape} and noise_map shape "
                    f"{noise_shape} must match"
                )

            if band.mask is not None:
                mask_shape = np.shape(band.mask)
                if len(mask_shape) != 2:
                    raise ValueError(f"band '{band.name}' mask must be 2D, got shape {mask_shape}")
                if mask_shape != image_shape:
                    raise ValueError(
                        f"band '{band.name}' image_data shape {image_shape} and mask shape "
                        f"{mask_shape} must match"
                    )

    def _build_band_models(self) -> tuple[ImageProbModel, ...]:
        band_models = []

        for idx, (band, phys_model) in enumerate(zip(self.bands, self.phys_models)):
            band_model = ImageProbModel(
                image_data=band.image_data,
                noise_map=band.noise_map,
                psf_kernel=band.psf_kernel,
                dpix=band.dpix,
                nsub=band.nsub,
                phys_model=phys_model,
                use_linear=self.use_linear,
                mask=band.mask,
                solver_type=self.solver_type,
            )
            attr_name = f"band_model_{idx}"
            setattr(self, attr_name, band_model)
            band_models.append(band_model)

        return tuple(band_models)

    @staticmethod
    def _normalize_param_value(value: Any) -> Any:
        if isinstance(value, dict):
            return {k: MultiBandImageProbModel._normalize_param_value(v) for k, v in value.items()}
        arr = np.asarray(value)
        if arr.shape == ():
            scalar = arr.item()
            if isinstance(scalar, dict):
                return {k: MultiBandImageProbModel._normalize_param_value(v) for k, v in scalar.items()}
            return scalar
        return value

    @staticmethod
    def _flatten_named_params(container: Any, out: Dict[str, Any]) -> None:
        if not isinstance(container, dict):
            return
        for key, value in container.items():
            if isinstance(value, dict):
                MultiBandImageProbModel._flatten_named_params(value, out)
            else:
                out[key] = MultiBandImageProbModel._normalize_param_value(value)

    @staticmethod
    def _sanitize_joint_loglike(log_like: Array) -> Array:
        return jnp.where(jnp.isfinite(log_like), log_like, -jnp.inf)

    @ck.forward
    def __call__(self) -> Array:
        band_loglikes = []
        for band_idx, band_model in enumerate(self.band_models):
            if self._band_identity_geometry[band_idx]:
                band_loglikes.append(band_model())
                continue

            band_loglikes.append(self._evaluate_non_identity_band_loglike(band_idx, band_model))

        joint_loglike = jnp.sum(jnp.stack(band_loglikes))
        return self._sanitize_joint_loglike(joint_loglike)

    def _evaluate_non_identity_band_loglike(self, band_idx: int, band_model: ImageProbModel) -> Array:
        xgrid_sub, ygrid_sub = self._build_transformed_subgrid_1d(band_idx, band_model)
        forward_result = cast(
            Tuple[Array, Array],
            band_model.forward_model(xgrid_sub=xgrid_sub, ygrid_sub=ygrid_sub),
        )
        image_model, intensity_list = forward_result
        return band_model._evaluate_loglike_from_forward_model(image_model, intensity_list)

    def _build_transformed_subgrid_1d(
        self,
        band_idx: int,
        band_model: ImageProbModel,
    ) -> tuple[Array, Array]:
        if self._band_identity_geometry[band_idx]:
            return band_model.sim_obj.xgrid_sub, band_model.sim_obj.ygrid_sub

        band = self.bands[band_idx]
        alignment_module = self._band_alignment_modules[band_idx]
        shift_x = alignment_module.shift_x.value
        shift_y = alignment_module.shift_y.value
        rotation = alignment_module.rotation.value

        if shift_x is None or shift_y is None or rotation is None:
            raise ValueError("alignment parameters must have concrete numeric values")

        xgrid_sub_2d, ygrid_sub_2d = make_grid_2d_transformed(
            npix=int(band.image_data.shape[0]),
            dpix=float(band.dpix),
            nsub=int(band.nsub),
            shift_x=jnp.asarray(shift_x),
            shift_y=jnp.asarray(shift_y),
            rotation=jnp.asarray(rotation),
        )

        flat_indices = band_model.sim_obj.sim_config.flat_indices
        xgrid_sub_1d = xgrid_sub_2d.reshape(-1)[flat_indices]
        ygrid_sub_1d = ygrid_sub_2d.reshape(-1)[flat_indices]
        return xgrid_sub_1d, ygrid_sub_1d

    @staticmethod
    def _get_band_linear_solved_params(
        band_model: ImageProbModel,
        intensity_list: Array,
    ) -> Dict[str, Any]:
        params = band_model.get_values("dict")
        params = {k: np.array(v) for k, v in params.items()}

        intensity_array = np.array(intensity_list)
        n_src = len(band_model.phys_model.source_light)
        n_lens = len(band_model.phys_model.lens_light)

        def update_intensity_param(module: Any, value: Any) -> None:
            for name in ["Ie", "amp", "intensity", "I0", "flux"]:
                if hasattr(module, name):
                    param = getattr(module, name)
                    params[param.name] = value
                    return

        for i in range(n_src):
            update_intensity_param(band_model.phys_model.source_light[i], intensity_array[i])

        for i in range(n_lens):
            update_intensity_param(band_model.phys_model.lens_light[i], intensity_array[n_src + i])

        return params

    def _solve_non_identity_band_linear_params(self, band_idx: int, band_model: ImageProbModel) -> Dict[str, Any]:
        xgrid_sub, ygrid_sub = self._build_transformed_subgrid_1d(band_idx, band_model)
        forward_result = cast(
            Tuple[Array, Array],
            band_model.forward_model(
                use_linear=True,
                return_intensity=True,
                xgrid_sub=xgrid_sub,
                ygrid_sub=ygrid_sub,
            ),
        )
        _, intensity_list = forward_result
        return self._get_band_linear_solved_params(band_model, intensity_list)

    def likelihood(self) -> float:
        return float(np.asarray(self.__call__()))

    def get_linear_solved_params(self, theta: Union[Sequence, Dict]) -> Dict[str, Dict[str, Any]]:
        self.set_values(theta)

        solved_by_band: Dict[str, Dict[str, Any]] = {}
        for band_idx, (band_name, band_model) in enumerate(zip(self.band_names, self.band_models)):
            if self._band_identity_geometry[band_idx]:
                raw_band_params = band_model.get_linear_solved_params({})
            else:
                raw_band_params = self._solve_non_identity_band_linear_params(band_idx, band_model)
            normalized_band_params: Dict[str, Any] = {}

            for param_name, param_value in raw_band_params.items():
                if param_name == "phys_model":
                    nested_params = self._normalize_param_value(param_value)
                    self._flatten_named_params(nested_params, normalized_band_params)
                else:
                    normalized_band_params[param_name] = self._normalize_param_value(param_value)

            solved_by_band[band_name] = normalized_band_params

        return solved_by_band

    def get_dynamic_params(self) -> list:
        return list(self.dynamic_params)
