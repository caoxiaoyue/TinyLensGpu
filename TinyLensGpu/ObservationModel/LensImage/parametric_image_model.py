"""
Probability model for gravitational lensing image fitting.

This module provides the probability model using PhysicalModel
and LensSimulator for computing image likelihoods.
"""

# pyright: reportMissingImports=false

import functools
import caskade as ck
import jax.numpy as jnp
import jax
from jax import jit, Array
import numpy as np
from typing import Optional, Dict, Tuple, Union, Sequence, Any, cast

from TinyLensGpu.ForwardSimulation.LensImage.parametric import LensSimulator
from TinyLensGpu.ForwardSimulation.LensImage.config import SimulatorConfig
from TinyLensGpu.PhysicalModel.LensImage.composite import PhysicalModel


class ImageProbModel(ck.Module):
    """
    Probability model for gravitational lensing images.

    This class computes the likelihood of model parameters given observed data,
    using a forward model (PhysicalModel + LensSimulator).

    Parameters
    ----------
    image_data : array_like
        Observed image data, shape (npix, npix)
    noise_map : array_like
        Noise map (standard deviations), shape (npix, npix)
    psf_kernel : array_like
        Point spread function kernel
    dpix : float
        Pixel scale in arcsec/pixel
    nsub : int
        Subsampling factor for ray-tracing
    phys_model : PhysicalModel
        Physical model containing mass and light components
    use_linear : bool
        Whether to use linear solver for intensity parameters
    mask : array_like, optional
        Boolean mask array (True = masked out)
    solver_type : str, optional
        Linear solver type: 'nnls' or 'normal' (default: 'nnls')
    position_likelihood : dict, optional
        Position likelihood constraint configuration

    Attributes
    ----------
    image_data : jnp.ndarray
        Observed image
    noise_map : jnp.ndarray
        Noise map
    sim_obj : LensSimulator
        lens simulator instance
    use_linear : bool
        Linear solver flag
    unmask : jnp.ndarray
        Boolean mask (True = unmasked pixels)
    position_like_config : dict
        Position likelihood configuration
    """

    def __init__(
        self,
        image_data: Union[np.ndarray, Array],
        noise_map: Union[np.ndarray, Array],
        psf_kernel: Union[np.ndarray, Array],
        dpix: float,
        nsub: int,
        phys_model: PhysicalModel,
        use_linear: bool,
        mask: Optional[Union[np.ndarray, Array]] = None,
        solver_type: str = 'nnls',
        position_likelihood: Optional[Dict] = None,
    ) -> None:
        """
        Initialize the ImageProbModel.

        Parameters
        ----------
        image_data : Union[np.ndarray, Array]
            Observed image data (npix, npix).
        noise_map : Union[np.ndarray, Array]
            Standard deviation of noise for each pixel (npix, npix).
        psf_kernel : Union[np.ndarray, Array]
            Point Spread Function kernel.
        dpix : float
            Pixel scale (arcseconds per pixel).
        nsub : int
            Subsampling factor for ray-tracing (e.g. 2 means 2x2 sub-pixels).
        phys_model : PhysicalModel
            Physical model containing mass and light profiles.
        use_linear : bool
            If True, use linear least squares to solve for intensity parameters.
        mask : Optional[Union[np.ndarray, Array]], optional
            Boolean mask where True indicates pixels to exclude from fitting, by default None.
        solver_type : str, optional
            Linear solver type ('nnls' or 'normal'), by default 'nnls'.
        position_likelihood : Optional[Dict], optional
            Configuration for position-based likelihood penalties (e.g. point source positions), by default None.
        """
        super().__init__("image_prob_model")

        self.image_data = jnp.array(image_data)
        noise_map_array = jnp.array(noise_map)
        if jnp.any(noise_map_array <= 0):
            raise ValueError("noise_map must contain only positive values (noise_map > 0)")
        self.noise_map = noise_map_array

        # Keep phys_model in the caskade module tree so @ck.forward can inject theta
        self.phys_model = phys_model

        # Create simulator configuration
        sim_config = SimulatorConfig(
            dpix=dpix,
            npix=image_data.shape[0],
            psf_kernel=psf_kernel,
            nsub=nsub,
            mask=mask,
        )

        # Create simulator
        self.sim_obj = LensSimulator(
            phys_model=self.phys_model,
            sim_config=sim_config,
            solver_type=solver_type,
        )

        self.use_linear = use_linear
        self.unmask = jnp.array(~sim_config.mask)
        self.position_like_config = position_likelihood

        # Precompute Gaussian log-likelihood constant term: -0.5 * sum(ln(2*pi*sigma^2))
        # only for unmasked pixels
        log_sigma_sq = 2 * jnp.log(self.noise_map)
        log_2pi = jnp.log(2 * jnp.pi)
        self.log_like_const = -0.5 * jnp.sum((log_sigma_sq + log_2pi) * self.unmask)

        # Initialize position likelihood for JIT
        self._init_position_likelihood(self.position_like_config)

        # Precompute static flags for JIT-friendly branching
        self._check_nnls = bool(self.use_linear and self.sim_obj.solver_type == "nnls")

    def _init_position_likelihood(self, config: Optional[Dict]) -> None:
        """
        Initialize optional point-source position penalty settings.

        Parameters
        ----------
        config : Optional[Dict]
            Configuration dictionary containing observed image positions,
            separation threshold, and penalty amplitude.
        """
        self._pos_px = None
        self._pos_py = None
        self._pos_thr = jnp.array(0.0, dtype=jnp.float32)
        self._pos_minl = jnp.array(0.0, dtype=jnp.float32)
        self._has_pos_penalty = False

        if config is not None:
            positions = config.get('positions', [])
            if positions is not None and len(positions) >= 2:
                self._pos_px = jnp.array([p[0] for p in positions], dtype=jnp.float32)
                self._pos_py = jnp.array([p[1] for p in positions], dtype=jnp.float32)
                self._pos_thr = jnp.array(
                    float(config.get('threshold_arcsec', config.get('position_threshold', 0.0))),
                    dtype=jnp.float32
                )
                self._pos_minl = jnp.array(
                    float(config.get('min_log_like', config.get('min_position_likelihood', 0.0))),
                    dtype=jnp.float32
                )
                self._has_pos_penalty = True

    def get_dynamic_params(self):
        """
        Return dynamic parameters exposed by the physical model.

        Returns
        -------
        Any
            Dynamic parameter container managed by caskade.
        """
        return self.phys_model.dynamic_params

    @ck.forward
    def forward_model(
        self,
        *,
        use_linear: Optional[bool] = None,
        return_intensity: bool = True,
        ret_each_plane: bool = False,
        image_map: Optional[Union[np.ndarray, Array]] = None,
        noise_map: Optional[Union[np.ndarray, Array]] = None,
        xgrid_sub: Optional[Union[np.ndarray, Array]] = None,
        ygrid_sub: Optional[Union[np.ndarray, Array]] = None,
        psf_kernel: Optional[Union[np.ndarray, Array]] = None,
    ) -> Union[
        Array,
        Tuple[Array, Optional[Array]],
        Tuple[Array, Array],
        Tuple[Array, Array, Optional[Array]],
    ]:
        """
        Run the forward model to generate a simulated image.

        This method integrates the physical model and simulator to produce
        the expected image, optionally solving for linear parameters.

        Parameters
        ----------
        use_linear : bool, optional
            Override the instance's linear solver setting.
        return_intensity : bool, optional
            If True, return the intensity vector (default: True).
        ret_each_plane : bool, optional
            If True, return separate images for source and lens planes (default: False).
        image_map : Array, optional
            Override image data (for linear solver).
        noise_map : Array, optional
            Override noise map (for linear solver).
        xgrid_sub : Array, optional
            Override subgrid X coordinates.
        ygrid_sub : Array, optional
            Override subgrid Y coordinates.
        psf_kernel : Array, optional
            Override PSF kernel.

        Returns
        -------
        Union[Array, Tuple[Array, ...]]
            The simulated image and optionally other components depending on flags.
        """
        linear_flag = self.use_linear if use_linear is None else use_linear

        if linear_flag:
            return self.sim_obj.simulate(  # type: ignore[reportArgumentType]
                use_linear=True,
                return_intensity=return_intensity,
                ret_each_plane=ret_each_plane,
                xgrid_sub=xgrid_sub,
                ygrid_sub=ygrid_sub,
                psf_kernel=psf_kernel,
                image_map=image_map if image_map is not None else self.image_data,
                noise_map=noise_map if noise_map is not None else self.noise_map,
            )
        else:
            return self.sim_obj.simulate(  # type: ignore[reportArgumentType]
                use_linear=False,
                return_intensity=return_intensity,
                ret_each_plane=ret_each_plane,
                xgrid_sub=xgrid_sub,
                ygrid_sub=ygrid_sub,
                psf_kernel=psf_kernel,
            )

    def _evaluate_loglike_from_forward_model(
        self,
        image_model: Array,
        intensity_list: Array,
    ) -> Array:
        """
        Evaluate log-likelihood from forward-model outputs.

        This internal helper centralizes the post-simulation likelihood math
        used by ``__call__`` while preserving masking, constant terms, solver
        validity checks, optional position penalties, and finite-value guards.
        """
        chi2_image = (image_model - self.image_data) ** 2 / self.noise_map ** 2
        chi2_image = chi2_image * self.unmask
        log_like = -0.5 * jnp.sum(chi2_image) + self.log_like_const

        # If NNLS is configured and returns any negative intensity, treat as invalid
        if self._check_nnls:
            ok = jnp.all(intensity_list >= 0.0)
            log_like = jnp.where(ok, log_like, -jnp.inf)

        # Add position likelihood penalty (JIT/vmap safe)
        if self._has_pos_penalty:
            log_like = log_like + self._position_likelihood_penalty_jax()

        # Guard against NaN/Inf by returning a very small loglikelihood
        log_like = jnp.where(jnp.isfinite(log_like), log_like, -1e10)
        return log_like

    @ck.forward
    @functools.partial(jit, static_argnums=(0,))
    def __call__(self):
        r"""
        Evaluate the log-likelihood of the model given the data.

        Computes $\ln P(d|\theta) = -\frac{1}{2} \chi^2 + C$.
        If `use_linear=True`, this includes the implicit marginalization or maximization over linear parameters.

        If `solver_type='nnls'`, a hard constraint is applied: if any linear parameter is negative,
        the likelihood is set to $-\infty$.

        Returns
        -------
        Array
            Scalar log-likelihood value.
        """
        forward_result = cast(Tuple[Array, Array], self.forward_model())
        image_model, intensity_list = forward_result
        return self._evaluate_loglike_from_forward_model(image_model, intensity_list)

    def likelihood(self, debug: bool = True) -> float:
        """
        Compute log-likelihood of current model parameters.

        With caskade, parameters are already set in the PhysicalModel via
        caskade's parameter management system, so we don't pass them here.

        Parameters
        ----------
        debug : bool, optional
            Whether to check for NaN/Inf values (default: True).
            Note: In the current JAX implementation, NaN/Inf checks are always performed.

        Returns
        -------
        log_like : float
            Log-likelihood value
        """
        like = float(np.asarray(self.__call__()))
        return like


    @functools.partial(jit, static_argnums=(0,))
    def _position_likelihood_penalty_jax(self) -> Array:
        r"""
        Evaluate soft penalty for inconsistent multiply imaged positions.

        Penalizes the model if ray-traced image positions do not map to the same source position.
        The penalty function is a smooth approximation of a step function (soft truncation).

        $Penalty = min\_log\_like \cdot (1 - \exp(-ratio))$
        where $ratio = \max(separation) / threshold$.

        Returns
        -------
        Array
            Log-likelihood penalty (<= 0).
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

    def get_linear_solved_params(self, nonlinear_params: Union[Sequence, Dict]) -> Dict[str, Any]:
        """
        Update model with non-linear parameters and return dictionary including linear-solved parameters.

        Useful for retrieving the optimal intensities (amplitudes) after a fitting step.

        Parameters
        ----------
        nonlinear_params : array_like or dict
            Non-linear parameters to set in the model.

        Returns
        -------
        params_dict : dict
            Dictionary containing both non-linear and linear-solved parameters.
            Keys are parameter names (e.g. 'source_light_0_amp').
        """
        # Set non-linear parameters
        self.set_values(nonlinear_params)

        # Run forward model to get linear intensities
        # forward_model returns (image, intensity_list) when return_intensity=True
        # Note: if ret_each_plane is True, it returns more values. 
        # But default is False.
        forward_result = cast(
            Tuple[Array, Array],
            self.forward_model(use_linear=True, return_intensity=True),
        )
        _, intensity_list = forward_result

        # Get all dynamic parameters as a dictionary
        # This converts JAX arrays to numpy/scalars if needed by get_values implementation
        # But usually it returns what's stored.
        params = self.get_values("dict")

        # Convert JAX arrays in params to numpy for easier handling
        params = {k: np.array(v) for k, v in params.items()}

        # Map solved intensities to their corresponding parameters
        # intensity_list is a JAX array
        intensity_list = np.array(intensity_list)

        n_src = len(self.phys_model.source_light)
        n_lens = len(self.phys_model.lens_light)

        def update_intensity_param(module, value):
            # Try common intensity parameter names
            """
            Update the intensity parameter of a light profile module.

            Searches for standard intensity attributes ('Ie', 'amp', 'intensity', 'I0', 'flux')
            and updates the corresponding entry in the `params` dictionary.
            """
            for name in ['Ie', 'amp', 'intensity', 'I0', 'flux']:
                if hasattr(module, name):
                    param = getattr(module, name)
                    # Use the parameter name as key
                    params[param.name] = value
                    return

        # Update source light intensities
        for i in range(n_src):
            update_intensity_param(self.phys_model.source_light[i], intensity_list[i])

        # Update lens light intensities
        for i in range(n_lens):
            update_intensity_param(self.phys_model.lens_light[i], intensity_list[n_src + i])

        return params

    def __repr__(self) -> str:
        """
        Return compact likelihood-model summary string.

        Returns
        -------
        str
            Text summary with image size, linear mode flag, and solver backend.
        """
        return (f"ImageProbModel("
                f"npix={self.image_data.shape[0]}, "
                f"use_linear={self.use_linear}, "
                f"solver={self.sim_obj.solver_type})")
