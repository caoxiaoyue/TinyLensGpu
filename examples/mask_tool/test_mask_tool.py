"""Exercise canvas events and FITS round-trips without a display server."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import json
from pathlib import Path
import numpy as np
import pytest
from astropy.io import fits
from matplotlib.backend_bases import KeyEvent, MouseButton, MouseEvent

from mask_tool import FitsMaskEditor, load_fits_image
import mask_tool


@pytest.fixture
def editor(tmp_path):
    data = np.arange(20 * 40, dtype=np.float64).reshape(20, 40)
    header = fits.Header({"CRPIX1": 12.5, "CRPIX2": 7.5, "BUNIT": "Jy"})
    result = FitsMaskEditor(
        data, output_path=tmp_path / "mask.fits", header=header, brush_size=1
    )
    result.fig.canvas.draw()
    yield result
    result.disconnect()
    plt.close(result.fig)


def mouse(
    editor, x, y, *, modifiers=(), name="motion_notify_event", button=None, buttons=None
):
    px, py = editor.ax.transData.transform((x, y))
    event = MouseEvent(
        name,
        editor.fig.canvas,
        px,
        py,
        button=button,
        modifiers=modifiers,
        buttons=buttons,
    )
    editor.fig.canvas.callbacks.process(name, event)


def test_hover_draw_continuous_erase_and_release(editor):
    mouse(editor, 2, 3)
    assert "x=2, y=3" in editor.pixel_text.get_text()
    assert "data[y, x]=122" in editor.pixel_text.get_text()
    assert not editor.mask.any()

    # Ctrl hover, no mouse button: sparse events must create a continuous line.
    mouse(editor, 2, 3, modifiers=("ctrl",))
    mouse(editor, 14, 3, modifiers=("control",))
    assert editor.mask[3, 2:15].all()
    assert editor.mask.sum() == 13

    # Shift wins when both modifiers are held, and switching modes must not
    # connect the previous paint position to the new erase position.
    mouse(editor, 7, 3, modifiers=("ctrl", "shift"))
    assert not editor.mask[3, 7]
    assert editor.mask[3, 8:15].all()
    mouse(editor, 9, 3, modifiers=("shift",))
    assert not editor.mask[3, 7:10].any()

    # Current mouse modifiers override stale keypress state.
    editor._pressed.add("ctrl")
    before = editor.mask.copy()
    mouse(editor, 20, 10)
    np.testing.assert_array_equal(editor.mask, before)


def test_stationary_modifier_key_events(editor):
    mouse(editor, 8, 5)
    px, py = editor.ax.transData.transform((8, 5))
    canvas = editor.fig.canvas
    for name, key in (("key_press_event", "control"), ("key_release_event", "control")):
        canvas.callbacks.process(name, KeyEvent(name, canvas, key, px, py))
    assert editor.mask[5, 8]
    assert not editor._pressed
    canvas.callbacks.process(
        "key_press_event", KeyEvent("key_press_event", canvas, "shift", px, py)
    )
    assert not editor.mask[5, 8]


def test_brush_size_clips_edges_and_does_not_wrap(editor):
    editor.brush_slider.set_val(5)
    mouse(editor, 0, 0, modifiers=("ctrl",))
    assert editor.brush_preview.radius == 2.5
    yy, xx = np.indices(editor.mask.shape)
    np.testing.assert_array_equal(editor.mask, xx**2 + yy**2 <= 2.5**2)
    assert not editor.mask[-1, -1]


def test_leaving_image_breaks_stroke_and_toolbar_blocks_paint(editor):
    mouse(editor, 2, 3, modifiers=("ctrl",))
    editor.fig.canvas.callbacks.process(
        "motion_notify_event",
        MouseEvent(
            "motion_notify_event", editor.fig.canvas, -1, -1, modifiers=("ctrl",)
        ),
    )
    mouse(editor, 14, 3, modifiers=("ctrl",))
    assert editor.mask.sum() == 2
    assert not editor.mask[3, 3:14].any()

    editor.ax.set_navigate_mode("PAN")
    mouse(editor, 20, 10, modifiers=("ctrl",))
    assert editor.mask.sum() == 2
    assert "data[y, x]=420" in editor.pixel_text.get_text()


def test_left_drag_fallback(editor):
    editor.mode_selector.set_active(1)
    mouse(editor, 3, 6, name="button_press_event", button=MouseButton.LEFT)
    mouse(editor, 12, 6, buttons={MouseButton.LEFT})
    mouse(editor, 12, 6, name="button_release_event", button=MouseButton.LEFT)
    assert editor.mask[6, 3:13].all()
    mouse(editor, 18, 6)
    assert not editor.mask[6, 18]

    editor.mode_selector.set_active(2)
    mouse(editor, 6, 6, name="button_press_event", button=MouseButton.LEFT)
    mouse(editor, 9, 6, buttons={MouseButton.LEFT})
    assert not editor.mask[6, 6:10].any()


def test_save_button_roundtrip_latest_mask_and_wcs(editor):
    mouse(editor, 30, 15, modifiers=("ctrl",))
    # Prove source scaling metadata cannot change the binary output.
    editor.header["BSCALE"] = 2.5
    editor.header["BZERO"] = 100.0
    editor.header["BLANK"] = 0
    editor.save_button._observers.process("clicked", None)
    with fits.open(editor.output_path, checksum=True) as hdul:
        image = hdul[0]
        assert image.data.dtype == np.uint8
        assert image.data.shape == (20, 40)
        np.testing.assert_array_equal(image.data.astype(bool), editor.mask)
        assert image.data[15, 30] == 1
        assert set(np.unique(image.data)) == {0, 1}
        assert image.header["CRPIX1"] == 12.5
        assert image.header["CRPIX2"] == 7.5
        assert image.header["BUNIT"] == "1"
        assert "BSCALE" not in image.header
        assert "BZERO" not in image.header
        assert "BLANK" not in image.header
        assert image.verify_checksum() == 1

    mouse(editor, 30, 15, modifiers=("shift",))
    mouse(editor, 4, 2, modifiers=("ctrl",))
    editor.save_button._observers.process("clicked", None)
    saved = fits.getdata(editor.output_path)
    assert saved[15, 30] == 0
    assert saved[2, 4] == 1
    assert saved.sum() == 1

    resumed = FitsMaskEditor(
        editor.data, initial_mask=fits.getdata(editor.output_path).astype(bool)
    )
    try:
        assert resumed.mask.dtype == bool
        np.testing.assert_array_equal(resumed.mask, editor.mask)
    finally:
        resumed.disconnect()
        plt.close(resumed.fig)


def test_save_path_and_file_errors(editor, tmp_path):
    output = tmp_path / "custom.fits"
    assert editor.save(output) == output
    assert editor.output_path == output
    editor.mask[2, 4] = True
    editor.save_button._observers.process("clicked", None)
    assert fits.getdata(output)[2, 4] == 1
    assert "custom.fits" in editor.output_text.get_text()
    assert "Saved: custom.fits" in editor.status_text.get_text()
    with pytest.raises(OSError):
        editor.save(overwrite=False)
    editor.output_path = tmp_path / "missing_directory" / "mask.fits"
    editor.save_button._observers.process("clicked", None)
    assert "Save failed" in editor.status_text.get_text()
    assert fits.getdata(output)[2, 4] == 1


def test_extension_selection_and_invalid_inputs(tmp_path):
    path = tmp_path / "extension.fits"
    fits.HDUList(
        [
            fits.PrimaryHDU(),
            fits.ImageHDU(np.ones((4, 9)), name="SCI"),
        ]
    ).writeto(path)
    for hdu in (None, 1, "SCI"):
        image, header = load_fits_image(path, hdu=hdu)
        editor = FitsMaskEditor(image, header=header)
        assert editor.mask.shape == (4, 9)
        editor.disconnect()
        plt.close(editor.fig)
    with pytest.raises(ValueError, match="same shape"):
        FitsMaskEditor(image, initial_mask=np.zeros((9, 4)))
    with pytest.raises(ValueError, match="0/1"):
        FitsMaskEditor(image, initial_mask=np.full((4, 9), 2))
    cube_path = tmp_path / "cube.fits"
    fits.writeto(cube_path, np.zeros((2, 4, 9)))
    with pytest.raises(ValueError, match="2-D"):
        load_fits_image(cube_path)
    for invalid in (
        path,
        str(path),
        np.zeros((2, 4, 9)),
        np.empty((0, 4)),
        np.ones((4, 9), dtype=complex),
        [["text"]],
    ):
        with pytest.raises(ValueError, match="2-D array"):
            FitsMaskEditor(invalid)


def test_clear_and_all_nonfinite_data(tmp_path):
    editor = FitsMaskEditor(
        np.full((4, 9), np.nan), initial_mask=np.ones((4, 9), dtype=bool)
    )
    try:
        assert editor.image.get_clim() == (0.0, 1.0)
        editor.clear_button._observers.process("clicked", None)
        assert not editor.mask.any()
        assert not editor.overlay.get_array().any()
    finally:
        editor.disconnect()
        plt.close(editor.fig)


def test_array_input_copy_and_default_output(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    image = np.arange(35).reshape(5, 7)
    initial_mask = np.zeros(image.shape, dtype=bool)
    initial_mask[2, 3] = True
    current = FitsMaskEditor(image, initial_mask=initial_mask, title="Residual image")
    try:
        image.fill(-1)
        initial_mask.fill(False)
        assert current.data[2, 3] == 17
        assert current.mask[2, 3]
        assert current.ax.get_title() == "Residual image"
        output = current.save()
        assert output == Path("mask.fits")
        np.testing.assert_array_equal(fits.getdata(output).astype(bool), current.mask)
    finally:
        current.disconnect()
        plt.close(current.fig)


def test_header_is_copied():
    header = fits.Header({"CRPIX1": 12.5})
    current = FitsMaskEditor(np.zeros((4, 9)), header=header)
    try:
        header["CRPIX1"] = 99
        assert current.header["CRPIX1"] == 12.5
    finally:
        current.disconnect()
        plt.close(current.fig)


def test_cli_reads_fits_before_creating_editor(tmp_path, monkeypatch):
    image_path = tmp_path / "image.fits"
    fits.writeto(image_path, np.arange(35).reshape(5, 7))
    mask_path = tmp_path / "existing_mask.fits"
    mask = np.zeros((5, 7), dtype=np.uint8)
    mask[2, 3] = 1
    fits.writeto(mask_path, mask)
    monkeypatch.setattr(
        "sys.argv", ["mask_tool.py", str(image_path), "--initial-mask", str(mask_path)]
    )
    monkeypatch.setattr(plt, "show", lambda: None)
    current = mask_tool.main()
    try:
        assert current.data[2, 3] == 17
        assert current.mask[2, 3]
        assert current.output_path == tmp_path / "mask.fits"
        assert current.ax.get_title() == "image.fits"
    finally:
        current.disconnect()
        plt.close(current.fig)
    monkeypatch.setattr(
        "sys.argv", ["mask_tool.py", str(image_path), "--output", str(image_path)]
    )
    with pytest.raises(SystemExit) as error:
        mask_tool.main()
    assert error.value.code == 2


@pytest.mark.parametrize("rerun_import_cell", [False, True])
def test_notebook_replaces_cached_editor(editor, monkeypatch, rerun_import_cell):
    """Rerunning notebook cells must replace a cached class with the old UI."""
    from matplotlib.widgets import TextBox

    class StaleMaskEditor(FitsMaskEditor):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.output_text.set_visible(False)
            self.path_box = TextBox(
                self.fig.add_axes([0.34, 0.17, 0.5, 0.038]),
                "Save path ",
                initial=str(self.output_path),
            )

    monkeypatch.setattr(mask_tool, "FitsMaskEditor", StaleMaskEditor)
    tool_dir = Path(__file__).parent.resolve()
    monkeypatch.chdir(tool_dir)
    notebook = json.loads((tool_dir / "mask_tool_demo.ipynb").read_text())
    namespace = {
        "FitsMaskEditor": StaleMaskEditor,
        "mask_tool": mask_tool,
        "Path": Path,
        "plt": plt,
        "display": lambda canvas: None,
        "IMAGE_PATH": editor.output_path.with_name("image.fits"),
        "OUTPUT_PATH": editor.output_path,
        "image": editor.data,
        "image_header": editor.header,
    }
    if rerun_import_cell:
        exec("".join(notebook["cells"][2]["source"]), namespace)
    try:
        exec("".join(notebook["cells"][4]["source"]), namespace)
        current = namespace["editor"]
        assert not hasattr(current, "path_box"), "Notebook reused the old Save path UI"
        assert current.output_text.get_visible()
    finally:
        if "editor" in namespace:
            namespace["editor"].disconnect()
            plt.close(namespace["editor"].fig)
