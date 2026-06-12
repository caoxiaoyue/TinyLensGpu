#%%
"""
Visualize the Euclid strong-lensing dataset shipped with this example.

Reads the FITS products under ``data/`` using the metadata in
``metadata.json`` and produces a single multi-panel quick-look figure that
shows the observed image, the noise map, the S/N map, the PSF, the
feature mask, and an image-with-mask overlay.

Run as a script::

    python show_data.py

The figure is written to ``data/preview.png`` and also shown on screen.
"""

import json
import os
from pathlib import Path

# Make the script runnable from any working directory by anchoring on its location.
os.chdir(Path(__file__).parent)

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits


# ------------------------------------------------------------------ #
# Paths and metadata
# ------------------------------------------------------------------ #
DATA_DIR = Path("data")
META_PATH = DATA_DIR.parent / "metadata.json"

with open(META_PATH, "r", encoding="utf-8") as fp:
    META = json.load(fp)

DPIX = float(META["pixel_scale_arcsec"])  # arcsec per pixel


# ------------------------------------------------------------------ #
# Loading helpers
# ------------------------------------------------------------------ #
def _load_fits(path: Path) -> np.ndarray:
    """Read a FITS file as a float32 ndarray."""
    return np.asarray(fits.getdata(str(path))).astype(np.float32)


def _load_optional_fits(path: Path) -> np.ndarray | None:
    """Read a FITS file if it exists, otherwise return None."""
    if not path.exists():
        return None
    return _load_fits(path)


# ------------------------------------------------------------------ #
# Image extent helper
# ------------------------------------------------------------------ #
def image_extent(npix: int) -> list[float]:
    """Return the matplotlib extent (arcsec) for a square image of side ``npix``."""
    half = npix * DPIX / 2.0
    return [-half, half, -half, half]


# ------------------------------------------------------------------ #
# Main visualization
# ------------------------------------------------------------------ #
def main() -> None:
    # Core products (always expected to be present)
    image = _load_fits(DATA_DIR / "image.fits")
    noise = _load_fits(DATA_DIR / "noise.fits")
    psf = _load_fits(DATA_DIR / "psf.fits")

    # Optional products
    feature_mask = None #_load_optional_fits(DATA_DIR / "feature_mask.fits")
    mask = _load_optional_fits(DATA_DIR / "mask.fits")

    # Derived quantities
    snr_map = image / np.maximum(noise, 1e-8)
    # Robust stretch for the image: a few-percent percentile cap keeps bright
    # lensed arcs from washing out the rest of the scene.
    vmax_image = float(np.nanpercentile(image, 99.5))
    vmax_noise = float(np.nanpercentile(noise, 99.5))
    vmax_snr = float(np.nanpercentile(snr_map, 99.5))

    npix_img = image.shape[0]
    extent_img = image_extent(npix_img)
    npix_psf = psf.shape[0]
    extent_psf = image_extent(npix_psf)

    # ------------------------------------------------------------------ #
    # Figure layout
    # ------------------------------------------------------------------ #
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    (ax_image, ax_noise, ax_snr), (ax_psf, ax_mask, ax_overlay) = axes

    # --- Image ---------------------------------------------------------- #
    im0 = ax_image.imshow(image, origin="lower", extent=extent_img, cmap="viridis",
                          vmin=0.0, vmax=vmax_image)
    ax_image.set_title("Image (VIS flux)")
    ax_image.set_xlabel("arcsec")
    ax_image.set_ylabel("arcsec")
    fig.colorbar(im0, ax=ax_image, fraction=0.046, pad=0.04, label="counts")

    # --- Noise ---------------------------------------------------------- #
    im1 = ax_noise.imshow(noise, origin="lower", extent=extent_img, cmap="magma",
                          vmin=0.0, vmax=vmax_noise)
    ax_noise.set_title("Noise RMS (VIS)")
    ax_noise.set_xlabel("arcsec")
    ax_noise.set_ylabel("arcsec")
    fig.colorbar(im1, ax=ax_noise, fraction=0.046, pad=0.04, label="RMS counts")

    # --- S/N map -------------------------------------------------------- #
    im2 = ax_snr.imshow(snr_map, origin="lower", extent=extent_img, cmap="viridis",
                         vmin=0.0, vmax=vmax_snr)
    ax_snr.set_title("S/N map = image / noise")
    ax_snr.set_xlabel("arcsec")
    ax_snr.set_ylabel("arcsec")
    fig.colorbar(im2, ax=ax_snr, fraction=0.046, pad=0.04, label="S/N")

    # --- PSF ------------------------------------------------------------ #
    im3 = ax_psf.imshow(psf, origin="lower", extent=extent_psf, cmap="viridis")
    ax_psf.set_title(f"PSF ({npix_psf}x{npix_psf}, {DPIX:.3f}\"/px)")
    ax_psf.set_xlabel("arcsec")
    ax_psf.set_ylabel("arcsec")
    fig.colorbar(im3, ax=ax_psf, fraction=0.046, pad=0.04, label="normalized")

    # --- Feature / regular mask panel ---------------------------------- #
    if feature_mask is not None or mask is not None:
        if feature_mask is not None:
            base = feature_mask.astype(np.float32)
            title = "Feature mask (1 = kept)"
            cmap = "gray"
        else:
            base = mask.astype(np.float32)
            title = "Mask (1 = masked out)"
            cmap = "gray_r"
        im4 = ax_mask.imshow(base, origin="lower", extent=extent_img, cmap=cmap,
                             vmin=0.0, vmax=1.0)
        ax_mask.set_title(title)
        ax_mask.set_xlabel("arcsec")
        ax_mask.set_ylabel("arcsec")
        fig.colorbar(im4, ax=ax_mask, fraction=0.046, pad=0.04)
    else:
        ax_mask.text(0.5, 0.5, "No mask available", ha="center", va="center",
                     transform=ax_mask.transAxes)
        ax_mask.set_xticks([])
        ax_mask.set_yticks([])

    # --- Image with mask overlay -------------------------------------- #
    im5 = ax_overlay.imshow(image, origin="lower", extent=extent_img, cmap="viridis",
                            vmin=0.0, vmax=vmax_image)
    if mask is not None:
        # Highlight masked-out pixels (mask == True) in red.
        masked_rgba = np.zeros((mask.shape[0], mask.shape[1], 4))
        masked_rgba[..., 0] = 1.0  # red
        masked_rgba[..., 3] = mask.astype(np.float32) * 0.5  # 50% alpha where masked
        ax_overlay.imshow(masked_rgba, origin="lower", extent=extent_img,
                          interpolation="nearest")
    if feature_mask is not None:
        # Outline feature pixels (kept) in cyan contour.
        ax_overlay.contour(feature_mask, levels=[0.5], colors="cyan",
                           linewidths=0.6, extent=extent_img)
    ax_overlay.set_title("Image + mask overlay")
    ax_overlay.set_xlabel("arcsec")
    ax_overlay.set_ylabel("arcsec")
    fig.colorbar(im5, ax=ax_overlay, fraction=0.046, pad=0.04, label="counts")

    fig.suptitle(
        f"Euclid preview  |  lens id = {META['lens_id']}  |  band = {META['band']}  |  "
        f"dpix = {DPIX:.3f}\"/px  |  image shape = {tuple(image.shape)}",
        fontsize=12,
    )
    plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))

    out_path = DATA_DIR / "preview.png"
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.show()
    print(f"[show_data] saved preview figure to {out_path}")


if __name__ == "__main__":
    main()

# %%
