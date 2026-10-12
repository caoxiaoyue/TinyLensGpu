# Interactive FITS Mask Editor

Display a 2-D image array with a colorbar and create a mask using the mouse.
The interface uses Matplotlib canvases, events, and widgets. Astropy handles
FITS I/O, and NumPy handles arrays. No TinyLensGpu or JAX imports are needed.

`FitsMaskEditor(image, ...)` accepts an array directly. Read FITS files outside
the editor, or supply a residual image, crop, or generated image. The editor
copies the image, initial mask, and optional header, so editing does not modify
the caller's arrays.

## Jupyter Notebook

Open `mask_tool_demo.ipynb` in the `tinylens_gpu` environment and run its cells
in order. The notebook reads
`examples/pix_src_demo_operator/pipe/subhalo_0946/data/image.fits` by default
and saves to `mask.fits` in the same directory when you click Save.
Change `IMAGE_PATH` to use another FITS image.

```bash
source ~/anaconda3/bin/activate
conda activate tinylens_gpu
python -m pip install ipympl
cd /home/xycao/workspace/tinylens_dev/TinyLensGpu/examples/mask_tool
jupyter lab mask_tool_demo.ipynb
```

Enable the interactive backend in the first code cell:

```ipython
%matplotlib widget
```

Then create the editor. This example assumes the working directory is the
tool directory:

```python
import matplotlib.pyplot as plt
from astropy.io import fits
from IPython.display import display
from mask_tool import FitsMaskEditor

# Read FITS and select the image HDU, such as ext=0 or ext="SCI".
image, image_header = fits.getdata("/path/to/image.fits", ext=0, header=True)

with plt.ioff():
    editor = FitsMaskEditor(
        image,
        output_path="/path/to/mask.fits",
        header=image_header,  # Optional: preserve the corresponding WCS.
        title="image.fits",   # Optional: set the image title.
        brush_size=5,         # Circular brush diameter in image pixels.
    )
display(editor.fig.canvas)
```

When running from the repository root, use
`from examples.mask_tool.mask_tool import FitsMaskEditor` instead.
Keep the `editor` variable to access the mask or save it from Python.

## Controls

| Action | Behavior |
| --- | --- |
| Hover over the image | Display `x`, `y`, the original `data[y, x]`, and the mask value |
| Hold **Ctrl** and move | Paint without holding a mouse button; the red overlay indicates mask=1 |
| Hold **Shift** and move | Erase to mask=0; Shift takes priority when both modifiers are held |
| `Diameter (px)` slider | Adjust the brush diameter; a circle previews its footprint |
| `Save to` | Show a short, read-only destination hint; the notebook prints the full path |
| `Save mask (overwrite)` | Save the current mask, overwriting the output file; display success or errors below |
| `Clear mask` | Clear the current mask; save again to update the file on disk |
| `Paint` / `Erase` modes | Paint or erase by dragging with the left mouse button if modifier keys are intercepted |
| `Keys` mode | Use the default Ctrl/Shift + mouse movement controls |
| Toolbar pan/zoom | Suspend painting until pan/zoom mode is turned off |

Set `OUTPUT_PATH` in the notebook and pass it as `output_path` to the constructor.
You can also call `editor.save("/path/to/new_mask.fits")` to save elsewhere.
After a successful write, the Save button uses that new destination.
Relative paths are resolved against Python's current working directory.
The interface shows the filename after saving; use `print(editor.output_path)`
to inspect the full path.

The display uses `origin="lower"`: `data[0, 0]` is at the bottom left,
`x` is the column index, and `y` is the row index. Both indices start at zero.
Saving preserves the image array's shape and row/column order without flipping
or transposing it. The brush snaps to the nearest pixel center, and strokes
are filled between mouse events to avoid gaps during fast movement.
Display limits default to the 1st and 99th percentiles of finite pixel values.
This affects only the display; hover readouts show the original values.
Use `vmin`, `vmax`, and `cmap` to customize the display.

Click the image first to give the notebook canvas keyboard focus.
Use `%matplotlib widget`; the static `%matplotlib inline` backend does not
support interaction. If Ctrl/Shift does not work in your browser, select
`Paint` or `Erase` and drag with the left mouse button.
Save before closing the figure or rerunning the creation cell; closing does
not save automatically. Use `plt.close(editor.fig)` to close an existing figure.

The notebook's import and figure creation cells reload the tool code to avoid
reusing an older interface from Python's import cache. The creation cell prints
`Tool code` to identify the loaded source file. After external notebook edits,
reopen the file to load the updated cells. Save the current mask before
creating a new editor.

## Mask Format and Resuming an Edit

`editor.mask` is a NumPy `bool` array with the same shape as the image:
`True` means marked, and `False` means unmarked.
FITS images do not directly store NumPy's boolean dtype, so output uses
`uint8` containing only **0 and 1**. This matches TinyLensGpu's convention
that `True` marks pixels excluded from fitting.

When supplied, `header` preserves coordinate/WCS metadata in the output.
Intensity scaling and units that do not apply to a mask are removed.
Without `header`, the output contains a standard FITS image header and mask
metadata. For cropped or transformed arrays, supply a header that corresponds
to the resulting array.

```python
from astropy.io import fits

# Save from Python; overwrite by default and return the output Path.
saved_path = editor.save("/path/to/mask.fits")

# Read the mask back as a boolean array.
mask = fits.getdata(saved_path).astype(bool)
assert mask.shape == editor.data.shape

# Read an existing mask before passing its array to resume editing.
initial_mask = fits.getdata("/path/to/mask.fits").astype(bool)
with plt.ioff():
    resumed_editor = FitsMaskEditor(
        image,
        initial_mask=initial_mask,
        output_path="/path/to/mask.fits",
        header=image_header,
    )
display(resumed_editor.fig.canvas)
```

`image` must be a nonempty, real 2-D array. Read a FITS file before passing its
data to the editor. For a 3-D data cube, select a 2-D slice first.
`initial_mask` must also be an array with the same shape, containing boolean
or 0/1 values.

The editor defaults to `mask.fits` in the current working directory.
Arrays do not carry a source file path, so the caller should set `output_path`
to a separate mask file. The output directory must already exist.

## Desktop Window

With an interactive Matplotlib desktop backend, run:

```bash
python mask_tool.py /path/to/image.fits --output /path/to/mask.fits --brush-size 7
python mask_tool.py /path/to/image.fits --hdu SCI --initial-mask /path/to/mask.fits
```

The command-line entry point accepts file paths, reads the image and optional
initial mask, then passes arrays to the editor. By default, it selects the
first 2-D image HDU and saves to `mask.fits` beside the input image.
It rejects an output path that would overwrite the input image.

## Verification

```bash
source ~/anaconda3/bin/activate
conda activate tinylens_gpu
cd /home/xycao/workspace/tinylens_dev/TinyLensGpu/examples/mask_tool
MPLBACKEND=Agg pytest test_mask_tool.py
```

See the [Matplotlib event documentation](https://matplotlib.org/stable/users/explain/figure/event_handling.html),
[ipympl documentation](https://matplotlib.org/ipympl/index.html), and
[Astropy FITS image documentation](https://docs.astropy.org/en/stable/io/fits/usage/image.html)
for details on events, notebook backends, and FITS images.
