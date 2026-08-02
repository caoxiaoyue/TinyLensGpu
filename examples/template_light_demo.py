# %%
"""
ImageTemplateLight demo.

Builds a small analytic galaxy template, evaluates the interpolated
brightness field of an ImageTemplateLight model on an image grid, and
saves a side-by-side comparison of the template and the interpolation.

Run from the repo root:

    python examples/template_light_demo.py
"""

import os

import jax.numpy as jnp
import matplotlib.pyplot as plt

from TinyLensGpu.PhysicalModel import ImageTemplateLight


def make_galaxy_template(n=8, pixel_size=0.05):
    """A smooth elliptical pseudo-galaxy template on an n x n grid."""
    x = (jnp.arange(n) - (n - 1) / 2) * pixel_size
    X, Y = jnp.meshgrid(x, x)
    R = jnp.sqrt((X / 0.13) ** 2 + (Y / 0.08) ** 2)
    return jnp.exp(-R) * jnp.cos(0.6 * jnp.arctan2(Y, X)) ** 2


def main():
    # Template parameters: 8x8 grid, 0.05 arcsec per pixel.
    n_pix = 8
    pixel_size = 0.05
    # Spatial extent of the template image (edges, not pixel centers).
    # For an n x n grid with pixel size p: span = n/2 * p.
    template_span = n_pix / 2 * pixel_size  # = 0.2 arcsec

    template = make_galaxy_template(n=n_pix, pixel_size=pixel_size)

    model = ImageTemplateLight(image=template, pixel_size=pixel_size, scale=10.0)
    model.scale.to_static()
    model.center_x.to_static()
    model.center_y.to_static()

    # Evaluate on a fine image grid spanning a bit beyond the template.
    eval_span = 0.5
    x = jnp.linspace(-eval_span, eval_span, 201)
    X, Y = jnp.meshgrid(x, x)
    brightness = model.light(X, Y)

    # A few arbitrary off-grid query points (linear interpolation).
    qx = jnp.array([-0.0317, 0.0123, 0.3])
    qy = jnp.array([0.0214, -0.0456, 0.3])
    qb = model.light(qx, qy)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6))
    # Left panel: show the raw template at its true physical extent [-0.2, 0.2].
    im0 = axes[0].imshow(template, origin="lower",
                         extent=[-template_span, template_span] * 2,
                         cmap="magma")
    axes[0].set_title(f"Template ({template.shape[0]}x{template.shape[0]}, "
                      f"pixel={pixel_size:.2f} arcsec)")
    axes[0].set_xlabel("arcsec")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    # Right panel: interpolated brightness evaluated on the [-0.5, 0.5] grid.
    im1 = axes[1].imshow(brightness, origin="lower",
                         extent=[-eval_span, eval_span] * 2,
                         cmap="magma")
    axes[1].set_title("Bilinear interpolation (x10 scale)")
    axes[1].set_xlabel("arcsec")
    axes[1].plot(qx, qy, "cx", markersize=7, label="query points")
    axes[1].legend(loc="upper right", fontsize=8)
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    fig.tight_layout()

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "template_light_demo_output.png")
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    print(f"comparison plot saved to {out_path}")

    print("\nArbitrary-position brightness (linear interpolation):")
    for xi, yi, bi in zip(qx, qy, qb):
        print(f"  x={float(xi):+.4f}  y={float(yi):+.4f}  ->  {float(bi):.4f}")

    print(f"\nGrid statistics: min={float(jnp.min(brightness)):.4f}  "
          f"max={float(jnp.max(brightness)):.4f}  "
          f"(template peak=1 x scale=10)")


if __name__ == "__main__":
    main()

# %%
