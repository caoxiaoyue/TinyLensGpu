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
    os.makedirs(path_dir, exist_ok=True)


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
        except Exception as e:  # noqa: BLE001
            logger.warning("Could not load mask file %s: %s", mask_path, e)

    return image_data, noise_map, psf_kernel, mask


def get_mask_bounding_box(
    mask: np.ndarray,
    npix: int,
    dpix: float,
    center_x: float = 0.0,
    center_y: float = 0.0,
    square: bool = True,
) -> Tuple[Optional[Tuple[float, float]], Optional[Tuple[float, float]]]:
    """
    Calculate the bounding box (in arcsec) for unmasked pixels.

    Parameters
    ----------
    mask : np.ndarray
        Boolean mask (True = masked out).
    npix : int
        Number of pixels per side.
    dpix : float
        Pixel scale (arcsec/pixel).
    center_x, center_y : float, optional
        Center of the image grid in arcsec, by default 0.0.
    square : bool, optional
        If True, force the bounding box to be square, by default True.

    Returns
    -------
    xlim, ylim : tuple of float or None
        Bounding box limits (min, max) for x and y. Returns (None, None) if all masked.
    """
    rows, cols = np.where(~mask)
    if len(rows) == 0:
        return None, None

    rmin, rmax = rows.min(), rows.max()
    cmin, cmax = cols.min(), cols.max()

    # Convert pixel coordinates to arcsec coordinates
    # Image grid is centered at (center_x, center_y)
    x0 = center_x - npix * dpix / 2.0
    y0 = center_y - npix * dpix / 2.0

    # Edge of pixels
    xmin = x0 + cmin * dpix
    xmax = x0 + (cmax + 1) * dpix
    ymin = y0 + rmin * dpix
    ymax = y0 + (rmax + 1) * dpix

    if square:
        width = xmax - xmin
        height = ymax - ymin
        side = max(width, height)

        xc = (xmin + xmax) / 2.0
        yc = (ymin + ymax) / 2.0

        xlim = (float(xc - side / 2.0), float(xc + side / 2.0))
        ylim = (float(yc - side / 2.0), float(yc + side / 2.0))
    else:
        xlim = (float(xmin), float(xmax))
        ylim = (float(ymin), float(ymax))

    return xlim, ylim
