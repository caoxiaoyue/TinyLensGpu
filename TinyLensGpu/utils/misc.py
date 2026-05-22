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


def arc_mask_from(
    snr_map: np.ndarray,
    threshold: float = 3.0,
    ignor_size: int = 25,
    ext_size: int = 5,
    close_size: int = 3,
) -> np.ndarray:
    """
    Generate a clean arc mask from an SNR map using robust morphological operations.

    Steps:
    1. Threshold the SNR map to create a initial binary mask.
    2. Remove small isolated islands (noise) using connected component analysis.
    3. Apply morphological closing to fill small internal holes and bridge tiny gaps.
    4. Dilate the mask to ensure full coverage of the lensed features.

    Parameters
    ----------
    snr_map : np.ndarray
        2D array of signal-to-noise ratio.
    threshold : float
        SNR threshold for masking.
    ignor_size : int
        Minimum number of connected pixels to keep.
    ext_size : int
        Size of the dilation kernel.
    close_size : int
        Size of the footprint for morphological closing (hole filling).

    Returns
    -------
    mask : np.ndarray
        The resulting boolean mask, where True indicates pixels to be EXCLUDED
        (i.e., pixels that do NOT belong to the arc).
    """
    from skimage import measure
    from skimage.morphology import closing, dilation

    # Step 1: Binary thresholding
    bool_map = snr_map > threshold

    # Step 2: Remove small islands
    labels = measure.label(bool_map)
    label_sizes = np.bincount(labels.ravel())

    signal_mask = np.copy(bool_map)
    for label_idx, size in enumerate(label_sizes):
        if label_idx > 0 and size < ignor_size:
            signal_mask[labels == label_idx] = False

    # Step 3: Morphological Closing
    # Fills small holes and bridges tiny gaps in curved arcs.
    # Uses a square footprint of size 'close_size'.
    if close_size > 0:
        signal_mask = closing(
            signal_mask, footprint=np.ones((close_size, close_size))
        )

    # Step 4: Dilation
    # Final expansion to ensure the mask safely covers all relevant signal.
    if ext_size > 0:
        signal_mask = dilation(
            signal_mask, footprint=np.ones((ext_size, ext_size))
        )

    # Invert the mask: TinyLensGpu convention uses True for EXCLUDED pixels.
    # The signal_mask identifies the arc (signal), so we return its inverse.
    return ~signal_mask


def generate_radial_basis_knots(
    dpix: float,
    center_x: float = 0.0,
    center_y: float = 0.0,
    n_sigmas: int = 20,
    log_rmin: float = -2.0,
    log_rmax: float = np.log10(3.0),
    arc_mask: Optional[np.ndarray] = None,
    mode: str = "bspline",
) -> np.ndarray:
    """
    Generate logarithmically spaced radial basis knots (sigmas).

    Two generation modes are supported:

    - ``"bspline"`` (default): prepends a knot at radius ``0.0`` so that
      B-spline bases can reconstruct flux at the exact centre.  The
      returned array has the form ``[0.0] + 10**logspace(...)``.
    - ``"mge"`` : returns pure log-spaced knots without the central zero
      knot, suitable for Multi-Gaussian Expansion (MGE) component widths.

    When ``arc_mask`` is provided, knots are placed in the inner and outer
    regions that avoid the masked annulus.

    Parameters
    ----------
    dpix : float
        Pixel scale in arcsec.
    center_x, center_y : float, optional
        Center of the lens in arcsec, by default 0.0.
    n_sigmas : int, optional
        Total number of knots (sigmas) to generate, by default 20.
    log_rmin, log_rmax : float, optional
        Log10 boundaries for the knot distribution, by default -2.0 and
        log10(3.0).
    arc_mask : np.ndarray, optional
        Boolean mask where True indicates pixels inside the lensed arc region
        to avoid. Defaults to None.
    mode : {"bspline", "mge"}, optional
        Knot generation mode. Default is ``"bspline"``.

    Returns
    -------
    np.ndarray
        Array of generated knots (sigmas).
    """
    # Effective number of log-spaced knots to generate (excluding the central zero knot)
    n_base = n_sigmas - 1 if mode == "bspline" else n_sigmas

    if arc_mask is None:
        sigmas = 10 ** np.linspace(log_rmin, log_rmax, n_base)
    else:
        npix = arc_mask.shape[0]
        x_1d = (np.arange(npix) - npix / 2.0 + 0.5) * dpix - center_x
        y_1d = (np.arange(npix) - npix / 2.0 + 0.5) * dpix - center_y
        x_arcsec, y_arcsec = np.meshgrid(x_1d, y_1d)
        r_arcsec = np.hypot(x_arcsec, y_arcsec)

        arc_r = r_arcsec[arc_mask]
        if arc_r.size > 0:
            r_in = float(np.min(arc_r))
            r_out = float(np.max(arc_r))
        else:
            r_in = r_out = np.nan

        if np.isnan(r_in):
            sigmas = 10 ** np.linspace(log_rmin, log_rmax, n_base)
        else:
            log_rin_clamped = min(max(np.log10(r_in), log_rmin), log_rmax)
            log_rout_clamped = min(max(np.log10(r_out), log_rmin), log_rmax)

            len_in = log_rin_clamped - log_rmin
            len_out = log_rmax - log_rout_clamped
            total_len = len_in + len_out

            if total_len > 0:
                n_in = int(round(n_base * len_in / total_len))
                n_out = n_base - n_in
                sigmas_in = (
                    np.logspace(log_rmin, log_rin_clamped, n_in)
                    if n_in > 0
                    else np.array([])
                )
                sigmas_out = (
                    np.logspace(log_rout_clamped, log_rmax, n_out)
                    if n_out > 0
                    else np.array([])
                )
                sigmas = np.concatenate([sigmas_in, sigmas_out])
            else:
                sigmas = 10 ** np.linspace(log_rmin, log_rmax, n_base)

    if mode == "bspline":
        return np.concatenate([np.array([0.0]), sigmas])
    return sigmas
