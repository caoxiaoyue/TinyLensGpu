"""
Entry point for Euclid Q1 strong-lensing modeling.

Sets XLA / threading environment variables (must happen before jax import),
changes to this directory, then delegates to the shared ``pipeline.main``.

Usage::

    python model.py            # full 5-stage pipeline
    python model.py --skip-done   # resume from cached output/stage_*.pkl
"""

import os
import sys
from pathlib import Path

# Environment variables must be set before any jax import.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

# Anchor on the script's own directory so data/, output/, metadata.json and
# pipeline.py are all resolved with the same relative paths the pipeline uses.
os.chdir(Path(__file__).parent)

# pipeline.py lives in this same directory.
if str(Path(__file__).parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent))

from pipeline import main  # noqa: E402

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--skip-done", action="store_true",
        help="Re-use cached posteriors in output/stage_*.pkl",
    )
    parser.add_argument(
        "--out-dir", default=None,
        help="Output directory relative to this script.",
    )
    parser.add_argument("--source-interpolation", choices=("bilinear", "cubic_hermite"))
    parser.add_argument("--evidence-logdet", choices=("overlap", "block"), default="overlap")
    parser.add_argument("--evidence-block-size", type=int, default=10)
    parser.add_argument("--vectorized-chunk-size", type=int)
    parser.add_argument("--seed", type=int, help="Nautilus seed for the pixelized stages.")
    main(**vars(parser.parse_args()))
