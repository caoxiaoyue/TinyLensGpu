"""Fit the Euclid VIS mock with cubic Hermite interpolation in Stages M0–M3.

M1 renders the reference arc with Hermite interpolation. M2 and M3 project
that arc onto the source grid with the Hermite stencil.
"""

import argparse
from pathlib import Path

from pix_src_fit import DATA_DIR, ROOT, main as fit_pixelized


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "output" / "pixelized_3rd_itp")
    fit_pixelized(**vars(parser.parse_args()), interpolation="cubic_hermite")
