"""
Entry point for SLACS lens J1627-0053 modeling.

Sets XLA / threading environment variables (must happen before jax import),
changes to the lens directory, then delegates to the shared pipeline.
"""

import os
import sys
from pathlib import Path

# Environment variables must be set before any jax import.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

os.chdir(Path(__file__).parent)

# pipeline.py lives in the parent directory (slacs_examples/).
_sys_path_inserted = str(Path(__file__).parent.parent)
if _sys_path_inserted not in sys.path:
    sys.path.insert(0, _sys_path_inserted)

from pipeline import main  # noqa: E402

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--skip-done", action="store_true",
        help="Re-use cached posteriors in output/stage_*.pkl",
    )
    args = parser.parse_args()
    main(skip_done=args.skip_done)
