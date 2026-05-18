import logging
from typing import Optional, Tuple

import numpy as np
from astropy.io import fits

MASKED_NOISE_VALUE = 1e8

logger = logging.getLogger(__name__)


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
    image_data = fits.getdata(image_path).astype("float32")
    noise_map = fits.getdata(noise_path).astype("float32")
    psf_kernel = fits.getdata(psf_path).astype("float32")

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


def generate_radial_basis_knots_with_mask(
    arc_mask: np.ndarray,
    dpix: float,
    center_x: float = 0.0,
    center_y: float = 0.0,
    n_sigmas: int = 20,
    log_min: float = -2.0,
    log_max: float = np.log10(3.0),
) -> np.ndarray:
    """
    Generate logarithmically spaced radial basis knots (sigmas) that avoid a masked annular region.

    Parameters
    ----------
    arc_mask : np.ndarray
        Boolean mask where True indicates pixels inside the lensed arc region
        to avoid.
    dpix : float
        Pixel scale in arcsec.
    center_x, center_y : float, optional
        Center of the lens in arcsec, by default 0.0.
    n_sigmas : int, optional
        Total number of knots (sigmas) to generate, by default 20.
    log_min, log_max : float, optional
        Log10 boundaries for the knot distribution, by default -2.0 and
        log10(3.0).

    Returns
    -------
    np.ndarray
        Array of generated knots (sigmas).
    """
    npix = arc_mask.shape[0]
    y, x = np.indices((npix, npix))
    x_arcsec = (x - npix / 2 + 0.5) * dpix - center_x
    y_arcsec = (y - npix / 2 + 0.5) * dpix - center_y
    r_arcsec = np.hypot(x_arcsec, y_arcsec)

    arc_r = r_arcsec[arc_mask]
    if arc_r.size > 0:
        r_in = float(np.min(arc_r))
        r_out = float(np.max(arc_r))
    else:
        r_in = r_out = np.nan

    if np.isnan(r_in):
        return 10 ** np.linspace(log_min, log_max, n_sigmas)

    log_rin = min(max(np.log10(r_in), log_min), log_max)
    log_rout = min(max(np.log10(r_out), log_min), log_max)

    len_in = log_rin - log_min
    len_out = log_max - log_rout
    total_len = len_in + len_out

    if total_len > 0:
        n_in = int(round(n_sigmas * len_in / total_len))
        n_out = n_sigmas - n_in
        sigmas_in = (
            np.logspace(log_min, log_rin, n_in) if n_in > 0 else np.array([])
        )
        sigmas_out = (
            np.logspace(log_rout, log_max, n_out) if n_out > 0 else np.array([])
        )
        return np.concatenate([sigmas_in, sigmas_out])
    else:
        return 10 ** np.linspace(log_min, log_max, n_sigmas)
