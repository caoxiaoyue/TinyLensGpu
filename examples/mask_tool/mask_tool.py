"""Matplotlib FITS mask editor, usable with ``%matplotlib widget``.

Hold Ctrl and move to paint; hold Shift and move to erase. ``mask[y, x]``
is True for marked pixels. FITS output stores the same mask as uint8 (0/1).
Only NumPy, Astropy and Matplotlib are required; no TinyLensGpu/JAX imports.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from matplotlib.backend_bases import MouseButton
from matplotlib.colors import ListedColormap
from matplotlib.patches import Circle
from matplotlib.widgets import Button, RadioButtons, Slider


def load_fits_image(path, hdu=None):
    """Read a 2-D FITS image and header for the command-line example.

    Without an explicit HDU, select the first 2-D image HDU. Reading is
    separate from FitsMaskEditor, which accepts arrays only.
    """
    with fits.open(path, memmap=False) as hdul:
        if hdu is None:
            images = (
                item
                for item in hdul
                if isinstance(item, (fits.PrimaryHDU, fits.ImageHDU, fits.CompImageHDU))
                and item.data is not None
                and item.data.ndim == 2
            )
            selected = next(images, None)
            if selected is None:
                raise ValueError(f"No 2-D image HDU found in {path}.")
        else:
            selected = hdul[hdu]
        data = np.array(selected.data, copy=True)
        header = selected.header.copy()
    if data.ndim != 2 or data.size == 0 or data.dtype.kind not in "buif":
        raise ValueError("The selected HDU must contain a nonempty real 2-D image.")
    return data, header


class FitsMaskEditor:
    """Edit a 2-D image array mask using Matplotlib widgets and canvas events.

    Parameters
    ----------
    image : array-like
        Nonempty real 2-D image. Copied on construction; never transposed or
        flipped for saving. Read FITS files before creating the editor.
    output_path : str or pathlib.Path, optional
        Defaults to ``mask.fits`` in the current working directory. The Save button
        overwrites this output so it always contains the latest mask.
    header : astropy.io.fits.Header, optional
        Copied FITS header for mask output, including WCS. Defaults to an empty
        header. For cropped/transformed arrays, supply corresponding metadata.
    title : str, optional
        Image panel title, defaults to ``Image``.
    brush_size : int, optional
        Circular brush diameter in image pixels, initially 5.
    initial_mask : array-like, optional
        Same-size boolean or binary 0/1 array to resume editing. Read saved FITS
        masks before passing them here.
    cmap : str, optional
        Image colormap; defaults to ``gray``.
    vmin, vmax : float, optional
        Display limits only. Defaults to the finite pixels' 1st/99th percentiles.

    Notes
    -----
    The default Keys mode paints/erases with modifier + mouse movement,
    without requiring a mouse button. Paint/Erase modes provide a left-drag
    fallback if a notebook/browser intercepts modifier keys. Shift takes
    priority over Ctrl. Toolbar pan/zoom temporarily disables drawing.
    """

    def __init__(
        self,
        image,
        output_path="mask.fits",
        *,
        header=None,
        title="Image",
        brush_size=5,
        initial_mask=None,
        cmap="gray",
        vmin=None,
        vmax=None,
    ):
        self.data = np.array(image, copy=True)
        if (
            self.data.ndim != 2
            or self.data.size == 0
            or self.data.dtype.kind not in "buif"
        ):
            raise ValueError(
                "image must be a nonempty real 2-D array, not a FITS path."
            )
        if header is not None and not isinstance(header, fits.Header):
            raise TypeError("header must be an astropy.io.fits.Header or None.")
        self.header = fits.Header() if header is None else header.copy()
        self.output_path = Path(output_path).expanduser()
        self.mask = self._read_mask(initial_mask)
        if int(brush_size) != brush_size or brush_size < 1:
            raise ValueError("brush_size must be a positive integer diameter.")
        self.brush_size = int(brush_size)
        self._pressed = set()
        self._left_down = False
        self._last_point = None
        self._last_action = None
        self._cursor = None

        finite = self.data[np.isfinite(self.data)]
        low, high = np.percentile(finite, [1, 99]) if finite.size else (0.0, 1.0)
        if low == high:
            low, high = low - 0.5, high + 0.5

        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        self.fig.subplots_adjust(left=0.09, right=0.89, bottom=0.32, top=0.88)
        self.image = self.ax.imshow(
            self.data,
            origin="lower",
            interpolation="nearest",
            cmap=cmap,
            vmin=low if vmin is None else vmin,
            vmax=high if vmax is None else vmax,
        )
        self.colorbar = self.fig.colorbar(self.image, ax=self.ax, pad=0.025)
        self.colorbar.set_label("Pixel value")
        self.ax.set(xlabel="x (column)", ylabel="y (row)")
        self.ax.set_title(title)
        self.overlay = self.ax.imshow(
            self.mask,
            origin="lower",
            interpolation="nearest",
            cmap=ListedColormap([(1, 0.2, 0.1, 0), (1, 0.2, 0.1, 0.45)]),
            vmin=0,
            vmax=1,
        )
        self.brush_preview = Circle(
            (0, 0),
            self.brush_size / 2,
            fill=False,
            color="cyan",
            lw=1.3,
            visible=False,
        )
        self.ax.add_patch(self.brush_preview)
        self.fig.text(
            0.09,
            0.96,
            "Hold Ctrl + move: paint   |   Hold Shift + move: erase",
            fontsize=11,
        )
        self.pixel_text = self.fig.text(
            0.09, 0.92, "Move over the image to inspect data[y, x].", fontsize=10
        )
        self.status_text = self.fig.text(0.09, 0.025, "Ready.", fontsize=9)

        self.brush_slider = Slider(
            self.fig.add_axes([0.23, 0.245, 0.61, 0.028]),
            "Diameter (px)",
            1,
            max(64, self.brush_size),
            valinit=self.brush_size,
            valstep=1,
            valfmt="%0.0f",
        )
        self.mode_selector = RadioButtons(
            self.fig.add_axes([0.09, 0.085, 0.13, 0.12]),
            ("Keys", "Paint", "Erase"),
            active=0,
        )
        self.output_text = self.fig.text(
            0.34,
            0.185,
            self._output_label(),
            fontsize=10,
        )
        self.save_button = Button(
            self.fig.add_axes([0.34, 0.09, 0.22, 0.05]), "Save mask (overwrite)"
        )
        self.clear_button = Button(
            self.fig.add_axes([0.62, 0.09, 0.22, 0.05]), "Clear mask"
        )
        self.brush_slider.on_changed(self._on_brush_change)
        self.mode_selector.on_clicked(self._on_mode_change)
        self.save_button.on_clicked(self._on_save)
        self.clear_button.on_clicked(self._on_clear)
        callbacks = {
            "motion_notify_event": self._on_motion,
            "button_press_event": self._on_button_press,
            "button_release_event": self._on_button_release,
            "key_press_event": self._on_key_press,
            "key_release_event": self._on_key_release,
            "axes_leave_event": self._on_axes_leave,
            "figure_leave_event": self._on_figure_leave,
            "close_event": self._on_close,
        }
        self._connections = [
            self.fig.canvas.mpl_connect(name, callback)
            for name, callback in callbacks.items()
        ]
        # Matplotlib keeps weak references to bound callbacks. Keep the editor
        # alive for as long as its figure is open, including in notebooks.
        self.fig._fits_mask_editor = self

    def _read_mask(self, initial_mask):
        if initial_mask is None:
            return np.zeros(self.data.shape, dtype=bool)
        mask = np.asarray(initial_mask)
        if mask.shape != self.data.shape:
            raise ValueError("initial_mask must have the same shape as the image.")
        if not np.all((mask == 0) | (mask == 1)):
            raise ValueError("initial_mask must contain only boolean or 0/1 values.")
        return mask.astype(bool, copy=True)

    def _output_label(self):
        """Show a short destination hint; full paths belong in notebook output."""
        parts = self.output_path.parts
        label = "/".join(parts[-3:])
        if len(parts) > 3:
            label = ".../" + label
        if len(label) > 48:
            label = ".../" + label[-44:]
        return f"Save to: {label}"

    @staticmethod
    def _modifiers(key):
        parts = (key or "").lower().replace("control", "ctrl").split("+")
        return {part for part in parts if part in {"ctrl", "shift"}}

    def _pixel(self, event):
        if event.inaxes is not self.ax or event.xdata is None or event.ydata is None:
            return None
        if not np.isfinite(event.xdata) or not np.isfinite(event.ydata):
            return None
        x, y = int(np.floor(event.xdata + 0.5)), int(np.floor(event.ydata + 0.5))
        ny, nx = self.data.shape
        return (x, y) if 0 <= x < nx and 0 <= y < ny else None

    def _navigation_active(self):
        toolbar = getattr(self.fig.canvas, "toolbar", None)
        return (
            bool(getattr(toolbar, "mode", ""))
            or self.ax.get_navigate_mode() is not None
        )

    def _action(self):
        if self._navigation_active():
            return None
        if "shift" in self._pressed:
            return False
        if "ctrl" in self._pressed:
            return True
        if self._left_down:
            if self.mode_selector.value_selected == "Paint":
                return True
            if self.mode_selector.value_selected == "Erase":
                return False
        return None

    def _reset_stroke(self):
        self._last_point = self._last_action = None

    def _paint_segment(self, start, end, value):
        """Rasterize a round stroke, filling gaps between sparse mouse events."""
        x0, y0 = start
        x1, y1 = end
        radius = self.brush_size / 2
        ny, nx = self.mask.shape
        xmin = max(0, int(np.floor(min(x0, x1) - radius)))
        xmax = min(nx, int(np.ceil(max(x0, x1) + radius)) + 1)
        ymin = max(0, int(np.floor(min(y0, y1) - radius)))
        ymax = min(ny, int(np.ceil(max(y0, y1) + radius)) + 1)
        yy, xx = np.ogrid[ymin:ymax, xmin:xmax]
        dx, dy = x1 - x0, y1 - y0
        length_sq = dx * dx + dy * dy
        t = (
            np.clip(((xx - x0) * dx + (yy - y0) * dy) / length_sq, 0, 1)
            if length_sq
            else 0
        )
        inside = (xx - x0 - t * dx) ** 2 + (yy - y0 - t * dy) ** 2 <= radius**2
        self.mask[ymin:ymax, xmin:xmax][inside] = value

    def _update_cursor(self, point):
        self._cursor = point
        x, y = point
        self.brush_preview.center = point
        self.brush_preview.set_visible(True)
        self.brush_preview.set_edgecolor("lime" if "shift" in self._pressed else "cyan")
        self.pixel_text.set_text(
            f"x={x}, y={y}   |   data[y, x]={self.data[y, x]:.10g}"
            f"   |   mask={int(self.mask[y, x])}"
        )

    def _apply_at(self, point):
        action = self._action()
        if action is None:
            self._reset_stroke()
        else:
            start = self._last_point if self._last_action == action else None
            self._paint_segment(point if start is None else start, point, action)
            self._last_point, self._last_action = point, action
            self.overlay.set_data(self.mask)
            self.status_text.set_text("Painting (1)." if action else "Erasing (0).")
        self._update_cursor(point)
        self.fig.canvas.draw_idle()

    def _sync_mouse_state(self, event):
        # Modern backends report actual modifiers, including an empty set.
        # Prefer these over the last keypress, which can be stale after focus
        # changes. Older Matplotlib releases only provide key / key events.
        modifiers = getattr(event, "modifiers", None)
        if modifiers is not None:
            self._pressed = self._modifiers("+".join(modifiers))
        elif event.key is not None:
            self._pressed = self._modifiers(event.key)
        buttons = getattr(event, "buttons", None)
        if buttons is not None:
            self._left_down = MouseButton.LEFT in buttons

    def _on_motion(self, event):
        self._sync_mouse_state(event)
        point = self._pixel(event)
        if point is None:
            self._leave_image()
            return
        self._apply_at(point)

    def _on_button_press(self, event):
        self._sync_mouse_state(event)
        point = self._pixel(event)
        if point is not None and event.button == MouseButton.LEFT:
            self._left_down = True
            self._reset_stroke()
            self._apply_at(point)

    def _on_button_release(self, event):
        if event.button == MouseButton.LEFT:
            self._left_down = False
            self._reset_stroke()

    def _on_key_press(self, event):
        self._pressed.update(self._modifiers(event.key))
        if event.key == "escape":
            self._pressed.clear()
            self._left_down = False
            self._reset_stroke()
        point = self._pixel(event)
        if point is not None and self._modifiers(event.key):
            self._apply_at(point)

    def _on_key_release(self, event):
        key = (event.key or "").lower().replace("control", "ctrl").split("+")[-1]
        self._pressed.discard(key)
        self._reset_stroke()
        if self._cursor is not None:
            self._update_cursor(self._cursor)
        self.fig.canvas.draw_idle()

    def _leave_image(self):
        self._cursor = None
        self._reset_stroke()
        self.brush_preview.set_visible(False)
        self.fig.canvas.draw_idle()

    def _on_axes_leave(self, event):
        if event.inaxes is self.ax:
            self._leave_image()

    def _on_figure_leave(self, event):
        self._pressed.clear()
        self._left_down = False
        self._leave_image()

    def _on_brush_change(self, value):
        self.brush_size = int(value)
        self.brush_preview.set_radius(self.brush_size / 2)
        self._reset_stroke()
        self.fig.canvas.draw_idle()

    def _on_mode_change(self, label):
        self._reset_stroke()
        self.status_text.set_text(
            "Use Ctrl/Shift + move."
            if label == "Keys"
            else f"{label}: left-click and drag."
        )
        self.fig.canvas.draw_idle()

    def _on_clear(self, event):
        self.mask.fill(False)
        self._reset_stroke()
        self.overlay.set_data(self.mask)
        if self._cursor is not None:
            self._update_cursor(self._cursor)
        self.status_text.set_text("Mask cleared. Save to update the FITS file.")
        self.fig.canvas.draw_idle()

    def save(self, path=None, *, overwrite=True):
        """Save the current binary mask, preserving image coordinates/WCS.

        Defaults to ``output_path``. An explicit ``path`` becomes the new
        destination for subsequent saves after a successful write.
        File errors propagate to Python
        callers; the Save button displays them in the figure's status line.
        """
        if path is None:
            path = self.output_path
        path = Path(path).expanduser()
        header = self.header.copy()
        # Source intensity/scaling/checksum metadata must not reinterpret 0/1.
        for key in (
            "BSCALE",
            "BZERO",
            "BLANK",
            "CHECKSUM",
            "DATASUM",
            "DATAMIN",
            "DATAMAX",
        ):
            header.remove(key, ignore_missing=True, remove_all=True)
        header["BUNIT"] = ("1", "Dimensionless binary mask")
        header["MASKVAL"] = (1, "1 = marked/excluded; 0 = unmarked")
        header.add_history("Mask edited with the Matplotlib FitsMaskEditor.")
        fits.PrimaryHDU(self.mask.astype(np.uint8), header=header).writeto(
            path, overwrite=overwrite, checksum=True
        )
        self.output_path = path
        self.output_text.set_text(self._output_label())
        self.status_text.set_text(
            f"Saved: {path.name} ({int(self.mask.sum())} marked pixels)"
        )
        self.fig.canvas.draw_idle()
        return path

    def _on_save(self, event):
        try:
            self.save()
        except (OSError, ValueError) as exc:
            self.status_text.set_text(f"Save failed: {exc}")
            self.fig.canvas.draw_idle()

    def disconnect(self):
        """Disconnect editor callbacks; the figure remains available for viewing."""
        for connection in self._connections:
            self.fig.canvas.mpl_disconnect(connection)
        self._connections.clear()
        for widget in (
            self.brush_slider,
            self.mode_selector,
            self.save_button,
            self.clear_button,
        ):
            widget.disconnect_events()
        self.fig._fits_mask_editor = None

    def _on_close(self, event):
        self.disconnect()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path, help="Input 2-D FITS image")
    parser.add_argument(
        "--output", type=Path, help="Output path (default: mask.fits beside image)"
    )
    parser.add_argument(
        "--hdu", help="Image HDU index or name (default: first 2-D image)"
    )
    parser.add_argument(
        "--brush-size", type=int, default=5, help="Brush diameter in pixels"
    )
    parser.add_argument("--initial-mask", type=Path, help="Resume from a 0/1 mask FITS")
    args = parser.parse_args()
    hdu = int(args.hdu) if args.hdu is not None and args.hdu.isdecimal() else args.hdu
    image_path = args.image.expanduser()
    output_path = (
        args.output.expanduser()
        if args.output is not None
        else image_path.with_name("mask.fits")
    )
    if output_path.resolve() == image_path.resolve() or (
        output_path.exists()
        and image_path.exists()
        and output_path.samefile(image_path)
    ):
        parser.error("The mask output must differ from the input image path.")
    image, header = load_fits_image(image_path, hdu)
    initial_mask = (
        load_fits_image(args.initial_mask.expanduser())[0]
        if args.initial_mask is not None
        else None
    )
    editor = FitsMaskEditor(
        image,
        output_path,
        header=header,
        title=image_path.name,
        brush_size=args.brush_size,
        initial_mask=initial_mask,
    )
    plt.show()
    return editor


if __name__ == "__main__":
    main()
