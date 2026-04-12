"""Probability model for pixelized source gravitational lensing image fitting."""

from __future__ import annotations

import functools
from typing import Dict, Optional, Union

import caskade as ck
import jax.numpy as jnp
import numpy as np
from jax import Array, jit

from TinyLensGpu.ForwardSimulation.LensImage.config import SimulatorConfig
from TinyLensGpu.ForwardSimulation.LensImage.pixelized import PixelizedLensSimulator
from TinyLensGpu.PhysicalModel.LensImage.Pixelized.config import IrregularGridConfig, RectangularGridConfig
from TinyLensGpu.PhysicalModel.LensImage.composite import PhysicalModel


class PixelizedImageProbModel(ck.Module):
    """
    Image probability model for pixelized source reconstruction.

    The model wraps :class:`PixelizedLensSimulator`, performs linear inversion
    of source intensities, and returns the marginalized log evidence with an
    optional point-source position penalty.

    Parameters
    ----------
    image_data : array_like
        Observed image with shape ``(npix, npix)``.
    noise_map : array_like
        Per-pixel 1-sigma noise map with same shape as ``image_data``.
    sim_config : SimulatorConfig
        Image size and pixel scale, mask, and PSF settings.
    phys_model : PhysicalModel
        Physical model containing a pixelized source component.
    lensed_source_image : array_like, optional
        Optional adaptive-grid guide image.
    position_likelihood : dict, optional
        Configuration of optional multi-image position penalty.
    """

    def __init__(
        self,
        image_data: Union[np.ndarray, Array],
        noise_map: Union[np.ndarray, Array],
        sim_config: SimulatorConfig,
        phys_model: PhysicalModel,
        lensed_source_image: Optional[Union[np.ndarray, Array]] = None,
        position_likelihood: Optional[Dict] = None,
    ) -> None:
        """
        Initialize a `PixelizedImageProbModel` instance.

        This model computes the marginal likelihood (Bayesian evidence) of the image data given the
        lens mass model and pixelized source regularization hyperparameters. The source surface
        brightness is marginalized out analytically (linear inversion).

        Parameters
        ----------
        image_data : Union[np.ndarray, Array]
            The observed imaging data (pixel values), typically in units of counts or flux.
            Shape must match `sim_config.npix` x `sim_config.npix`.
        noise_map : Union[np.ndarray, Array]
            The 1-sigma noise map corresponding to `image_data`. Must have the same shape.
        sim_config : SimulatorConfig
            Configuration object defining the simulation grid (e.g., pixel count, pixel scale, mask).
        phys_model : PhysicalModel
            The physical model definition containing lens mass and source light components.
            Must contain a pixelized source model.
        lensed_source_image : Optional[Union[np.ndarray, Array]], optional
            An optional pre-computed image of the lensed source. If provided, it can be used
            for certain optimization or debugging workflows. Defaults to None.
        position_likelihood : Optional[Dict], optional
            Configuration dictionary for adding a penalty based on the positions of multiply
            imaged point sources (e.g., quasars). Defaults to None.

        Raises
        ------
        ValueError
            If the shapes of `image_data`, `noise_map`, or `mask` do not match `sim_config`.
        """
        super().__init__("pixelized_image_prob_model")

        self.image_data = jnp.asarray(image_data)
        self.noise_map = jnp.asarray(noise_map)
        self.sim_config = sim_config
        self.phys_model = phys_model
        extracted_pix_src_model = self.phys_model.get_pixelized_source_model()
        object.__setattr__(self, "pix_src_model", extracted_pix_src_model) #avoid it to be registered as a caskade submodule

        self.npix = int(self.sim_config.npix)
        expected_shape = (self.npix, self.npix)

        if self.image_data.shape != expected_shape:
            raise ValueError(
                f"image_data shape mismatch: expected {expected_shape}, got {self.image_data.shape}."
            )
        if self.noise_map.shape != expected_shape:
            raise ValueError(
                f"noise_map shape mismatch: expected {expected_shape}, got {self.noise_map.shape}."
            )

        self.mask = jnp.asarray(self.sim_config.mask, dtype=bool)
        if self.mask.shape != expected_shape:
            raise ValueError(
                f"sim_config.mask shape mismatch: expected {expected_shape}, got {self.mask.shape}."
            )

        self.lensed_source_image = lensed_source_image
        if self.lensed_source_image is not None and np.asarray(self.lensed_source_image).shape != expected_shape:
            raise ValueError(
                "lensed_source_image shape mismatch: "
                f"expected {expected_shape}, got {np.asarray(self.lensed_source_image).shape}."
            )

        self.unmask = ~self.mask
        self._data_vector = self.image_data[self.unmask]
        self._noise_variance = self.noise_map[self.unmask] ** 2

        self.position_like_config = position_likelihood
        self._init_position_likelihood(self.position_like_config)

        self.simulator = PixelizedLensSimulator(
            phys_model=self.phys_model,
            sim_config=self.sim_config,
            lensed_source_image=(
                None if self.lensed_source_image is None else np.asarray(self.lensed_source_image)
            ),
        )

    def _init_position_likelihood(self, config: Optional[Dict]) -> None:
        """
        Initialize the position likelihood penalty configuration.

        This sets up the parameters for penalizing the lens model if the predicted positions
        of multiply imaged point sources deviate too much from their observed positions or
        do not map back to a common source position within a threshold.

        Parameters
        ----------
        config : Optional[Dict]
            A dictionary containing configuration keys:
            - 'positions': List of [x, y] coordinates of observed image positions.
            - 'threshold_arcsec' or 'position_threshold': Max allowed source plane spread (arcsec).
            - 'min_log_like' or 'min_position_likelihood': Penalty value applied when threshold is exceeded.
        """
        self._pos_px = None
        self._pos_py = None
        self._pos_thr = jnp.array(0.0, dtype=jnp.float32)
        self._pos_minl = jnp.array(0.0, dtype=jnp.float32)
        self._has_pos_penalty = False

        if not config:
            return

        positions = config.get("positions", [])
        if positions is None or len(positions) < 2:
            return

        self._pos_px = jnp.array([p[0] for p in positions], dtype=jnp.float32)
        self._pos_py = jnp.array([p[1] for p in positions], dtype=jnp.float32)

        def get_val(keys, default):
            """
            Return first available value among alias keys.

            Parameters
            ----------
            keys : Sequence[str]
                Candidate key names searched in ``config``.
            default : Any
                Value returned when no key is found.

            Returns
            -------
            Any
                Retrieved value or ``default``.
            """
            for key in keys:
                if key in config:
                    return config[key]
            return default

        threshold = get_val(["threshold_arcsec", "position_threshold"], 0.0)
        min_log_like = get_val(["min_log_like", "min_position_likelihood"], 0.0)

        self._pos_thr = jnp.array(float(threshold), dtype=jnp.float32)
        self._pos_minl = jnp.array(float(min_log_like), dtype=jnp.float32)
        self._has_pos_penalty = True

    @ck.forward
    def _build_inverter(self):
        """
        Construct the linear inversion solver for the current model state.

        This method initializes the `LinearInversion` or `NNLSInversion` object (via the simulator)
        using the current data vector, noise variance, and regularization parameters.

        Returns
        -------
        inverter : LinearInversion
            An initialized inverter object capable of computing the Bayesian evidence
            and reconstructing the source.
        """
        result = self.simulator.forward(
            data=self.image_data,
            noise_map=self.noise_map,
            return_solver=True,
            return_image_2d=False,
            _solver_only=True,
        )
        inverter = result.inverter
        if inverter is None:
            raise RuntimeError("PixelizedLensSimulator.forward() did not return an inverter.")
        return inverter

    @ck.forward
    def __call__(self):
        """
        Compute the total log-probability (evidence + penalties) of the model.

        Calculates the Bayesian evidence of the linear inversion and adds any
        position likelihood penalties if configured.

        Returns
        -------
        log_prob : Array
            The total log-probability value. Returns -inf if the evidence is not finite.
        """
        inverter = self._build_inverter()
        log_ev = inverter.log_evidence()
        if self._has_pos_penalty:
            log_ev = log_ev + self._position_likelihood_penalty_jax()
        return jnp.where(jnp.isfinite(log_ev), log_ev, -jnp.inf)

    def log_evidence(self) -> float:
        """
        Evaluate scalar Bayesian evidence for current parameters.

        Returns
        -------
        float
            Marginal log evidence value.
        """
        return float(np.asarray(self.__call__()))

    @functools.partial(jit, static_argnums=(0,))
    def _position_likelihood_penalty_jax(self) -> Array:
        """
        Evaluate position-likelihood penalty using JAX.

        Calculates the ray-traced source positions for the observed image positions.
        If the spread of these source positions exceeds `_pos_thr`, a penalty is applied.

        Returns
        -------
        penalty : Array
            The calculated log-likelihood penalty (negative value).
        """
        beta_x, beta_y = self.phys_model.deflection(self._pos_px, self._pos_py)

        dx = beta_x[:, None] - beta_x[None, :]
        dy = beta_y[:, None] - beta_y[None, :]
        dist = jnp.sqrt(dx * dx + dy * dy)
        max_sep = jnp.max(dist)

        exceed = jnp.maximum(0.0, max_sep - self._pos_thr)
        ratio = jnp.where(self._pos_thr > 0.0, exceed / self._pos_thr, 0.0)
        pen_continuous = self._pos_minl * (1.0 - jnp.exp(-ratio))

        pen_clipped = jnp.clip(pen_continuous, min=self._pos_minl, max=0.0)
        return pen_clipped

    def __repr__(self) -> str:
        """
        Return a string representation of the `PixelizedImageProbModel`.

        Returns
        -------
        str
            A string summarizing the model configuration, including pixel count
            and source grid type.
        """
        if isinstance(self.pix_src_model.grid, IrregularGridConfig):
            n_source_points = int(self.pix_src_model.grid.n_source_points)
        elif isinstance(self.pix_src_model.grid, RectangularGridConfig):
            n_source_points = int(self.pix_src_model.grid.nx * self.pix_src_model.grid.ny)
        else:
            n_source_points = 0

        return (
            "PixelizedImageProbModel("
            f"npix={self.npix}, "
            f"n_source_points={n_source_points}, "
            f"source_grid_type='{self.pix_src_model.source_grid_type}')"
        )
