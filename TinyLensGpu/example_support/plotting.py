"""Shared image-panel drawing for example diagnostic figures."""

from __future__ import annotations


def draw_image_panel(
    ax, image, *, extent, title, xlabel="arcsec", ylabel=None,
    title_size=11, colorbar_fraction=0.046, colorbar_pad=0.04,
    colorbar_kwargs=None, **imshow_kwargs,
):
    """Draw one imshow panel with title/labels/colorbar; return the artist."""
    import matplotlib.pyplot as plt

    image_artist = ax.imshow(image, origin="lower", extent=extent, **imshow_kwargs)
    ax.set_title(title, fontsize=title_size)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    plt.colorbar(
        image_artist, ax=ax, fraction=colorbar_fraction, pad=colorbar_pad,
        **(colorbar_kwargs or {}),
    )
    return image_artist
