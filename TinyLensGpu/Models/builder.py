"""
Programmatic model builder for gravitational lensing (no YAML).

This module provides functions to build lens models programmatically,
following the example_v4.py style with direct parameter specification.
"""

from typing import List, Optional, Tuple, TYPE_CHECKING
import numpy as np
from astropy.io import fits
import logging

# Constants
MASKED_NOISE_VALUE = 1e8  # Large value to effectively mask pixels

logger = logging.getLogger(__name__)

from .param_u import ParamU
from .mass import SIE, Shear
from .light import SersicEllipse, GaussianEllipse
from .composite import PhysicalModel
import caskade as ck

if TYPE_CHECKING:
    from ..ProbModel.Image.image_model import ImageProbModel


def build_lens_model(
    lens_mass: Optional[List] = None,
    source_light: Optional[List] = None,
    lens_light: Optional[List] = None,
) -> PhysicalModel:
    """
    Build a physical model from component lists.
    
    Parameters
    ----------
    lens_mass : list, optional
        List of mass profile modules (e.g., SIE, Shear)
    source_light : list, optional
        List of source light profile modules
    lens_light : list, optional
        List of lens light profile modules
    
    Returns
    -------
    PhysicalModel
        Composite physical model
    
    Raises
    ------
    TypeError
        If any input is not a list or contains non-Module instances
    
    Examples
    --------
    >>> # Create components with ParamU parameters
    >>> sie = SIE(
    ...     theta_E=ParamU("theta_E", 1.5, prior_type="uniform", prior_settings=[0.5, 2.5]),
    ...     e1=ParamU("e1", 0.0, prior_type="gaussian", prior_settings=[0.0, 0.1]),
    ...     e2=ParamU("e2", 0.0, prior_type="gaussian", prior_settings=[0.0, 0.1]),
    ...     center_x=ParamU("center_x", 0.0),
    ...     center_y=ParamU("center_y", 0.0)
    ... )
    >>> 
    >>> # Set parameters to dynamic for sampling
    >>> sie.theta_E.to_dynamic()
    >>> sie.e1.to_dynamic()
    >>> sie.e2.to_dynamic()
    >>> 
    >>> # Build model
    >>> model = build_lens_model(lens_mass=[sie])
    """
    # Validate inputs
    for components, name in [
        (lens_mass, 'lens_mass'),
        (source_light, 'source_light'),
        (lens_light, 'lens_light')
    ]:
        if components is not None:
            if not isinstance(components, list):
                raise TypeError(
                    f"{name} must be a list, got {type(components).__name__}"
                )
            for i, comp in enumerate(components):
                if not isinstance(comp, ck.Module):
                    raise TypeError(
                        f"{name}[{i}] must be a ck.Module instance, "
                        f"got {type(comp).__name__}"
                    )
    
    return PhysicalModel(
        lens_mass=lens_mass or [],
        source_light=source_light or [],
        lens_light=lens_light or []
    )


def build_likelihood(
    phys_model: PhysicalModel,
    image_data: np.ndarray,
    noise_map: np.ndarray,
    psf_kernel: np.ndarray,
    pixel_scale: float,
    nsub: int = 4,
    use_linear: bool = False,
    mask: Optional[np.ndarray] = None,
    solver_type: str = 'nnls',
    position_likelihood: Optional[dict] = None,
) -> "ImageProbModel":
    """
    Build likelihood model from physical model and data.
    
    Parameters
    ----------
    phys_model : PhysicalModel
        Physical model with lens and light components
    image_data : np.ndarray
        Observed image data
    noise_map : np.ndarray
        Noise map (standard deviations)
    psf_kernel : np.ndarray
        Point spread function kernel
    pixel_scale : float
        Pixel scale in arcsec/pixel
    nsub : int, optional
        Subsampling factor for ray-tracing (default: 4)
    use_linear : bool, optional
        Whether to use linear solver for intensity parameters (default: False)
    mask : np.ndarray, optional
        Boolean mask array (True = masked out)
    solver_type : str, optional
        Linear solver type: 'nnls' or 'normal' (default: 'nnls')
    position_likelihood : dict, optional
        Position likelihood constraint configuration with keys:
        - 'positions': list of [x, y] positions in arcsec
        - 'threshold_arcsec': separation threshold
        - 'min_log_like': floor value when threshold violated
    
    Returns
    -------
    ImageProbModel
        Probability model for computing likelihoods
    
    Examples
    --------
    >>> # Load data
    >>> image = fits.getdata("image.fits")
    >>> noise = fits.getdata("noise.fits")
    >>> psf = fits.getdata("psf.fits")
    >>> 
    >>> # Build likelihood
    >>> likelihood = build_likelihood(
    ...     phys_model=model,
    ...     image_data=image,
    ...     noise_map=noise,
    ...     psf_kernel=psf,
    ...     pixel_scale=0.074
    ... )
    """
    # Lazy import to avoid circular dependency
    from ..ProbModel.Image.image_model import ImageProbModel
    
    return ImageProbModel(
        image_data=image_data,
        noise_map=noise_map,
        psf_kernel=psf_kernel,
        dpix=pixel_scale,
        nsub=nsub,
        phys_model=phys_model,
        use_linear=use_linear,
        mask=mask,
        solver_type=solver_type,
        position_likelihood=position_likelihood,
    )


def load_lens_data(
    image_path: str,
    noise_path: str,
    psf_path: str,
    mask_path: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Load lens imaging data from FITS files.
    
    Parameters
    ----------
    image_path : str
        Path to image FITS file
    noise_path : str
        Path to noise map FITS file
    psf_path : str
        Path to PSF kernel FITS file
    mask_path : str, optional
        Path to mask FITS file
    
    Returns
    -------
    image_data : np.ndarray
        Image data
    noise_map : np.ndarray
        Noise map
    psf_kernel : np.ndarray
        PSF kernel
    mask : np.ndarray or None
        Mask array (True = masked out) or None
    
    Examples
    --------
    >>> image, noise, psf, mask = load_lens_data(
    ...     "data/image.fits",
    ...     "data/noise.fits",
    ...     "data/psf.fits"
    ... )
    """
    image_data = fits.getdata(image_path).astype('float64')
    noise_map = fits.getdata(noise_path).astype('float64')
    psf_kernel = fits.getdata(psf_path).astype('float64')
    
    mask = None
    if mask_path is not None:
        try:
            mask = fits.getdata(mask_path).astype('bool')
            # Mask out noise
            noise_map = np.where(mask, MASKED_NOISE_VALUE, noise_map)
        except FileNotFoundError:
            logger.warning(f"Mask file not found: {mask_path}")
        except (OSError, IOError) as e:
            logger.warning(f"Could not load mask file {mask_path}: {e}")
        except Exception as e:
            logger.warning(f"Unexpected error loading mask file {mask_path}: {e}")
    
    return image_data, noise_map, psf_kernel, mask
