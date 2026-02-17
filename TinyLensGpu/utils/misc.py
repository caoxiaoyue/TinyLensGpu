import logging
import os
from typing import Optional, Tuple

import numpy as np
from astropy.io import fits

MASKED_NOISE_VALUE = 1e8

logger = logging.getLogger(__name__)


def auto_mkdir_path(path_dir: str) -> None:
    """
    Create a directory if it does not already exist.

    Parameters
    ----------
    path_dir : str
        Directory path to create.
    """
    if not os.path.exists(path_dir):
        abs_path = os.path.abspath(path_dir)
        os.makedirs(abs_path, exist_ok=True)


def load_lens_data(
    image_path: str,
    noise_path: str,
    psf_path: str,
    mask_path: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Load lens imaging data from FITS files.

    Returns
    -------
    image_data : np.ndarray
        Image data
    noise_map : np.ndarray
        Noise map (with optional masking applied)
    psf_kernel : np.ndarray
        PSF kernel
    mask : np.ndarray or None
        Boolean mask (True = masked out) if provided
    """
    image_data = fits.getdata(image_path).astype("float64")
    noise_map = fits.getdata(noise_path).astype("float64")
    psf_kernel = fits.getdata(psf_path).astype("float64")

    mask = None
    if mask_path is not None:
        try:
            mask = fits.getdata(mask_path).astype("bool")
            noise_map = np.where(mask, MASKED_NOISE_VALUE, noise_map)
        except FileNotFoundError:
            logger.warning("Mask file not found: %s", mask_path)
        except (OSError, IOError) as e:
            logger.warning("Could not load mask file %s: %s", mask_path, e)
        except Exception as e:  # noqa: BLE001
            logger.warning("Unexpected error loading mask file %s: %s", mask_path, e)

    return image_data, noise_map, psf_kernel, mask
