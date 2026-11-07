"""
Shared entry point for all SLACS lens modeling subdirectories.

Sets XLA / threading environment variables (must happen before jax import),
changes to the requested lens directory, then delegates to the shared pipeline.

Usage::

    # From slacs_examples/
    python model.py J0737+3216
    python model.py J1627-0053 --skip-done
    python model.py J0737+3216 --out-dir output_hermite --source-interpolation cubic_hermite

    # From inside a lens directory
    python ../model.py
    python ../model.py --skip-done

    # Absolute path also works
    python model.py /path/to/J0737+3216
"""

import os
import sys
from pathlib import Path

# Environment variables must be set before any jax import.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")


def main():
    import argparse
    import pickle

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "lens_dir",
        nargs="?",
        default=".",
        help="Lens subdirectory to run (default: current directory).",
    )
    parser.add_argument(
        "--skip-done",
        action="store_true",
        help="Re-use cached posteriors in output/stage_*.pkl",
    )
    parser.add_argument(
        "--out-dir", default=None,
        help="Output directory relative to the lens subdirectory.",
    )
    parser.add_argument(
        "--source-interpolation", choices=("bilinear", "cubic_hermite"),
        help="Fresh runs default to bilinear. Resumed runs use the checkpoint choice.",
    )
    parser.add_argument(
        "--initial-source-from", type=Path,
        help="Initialize a fresh M0 solve with S1 node intensities from a stage_m1.pkl.",
    )
    args = parser.parse_args()
    if args.skip_done and args.initial_source_from is not None:
        parser.error("--initial-source-from is only for a fresh run")

    lens_path = Path(args.lens_dir).resolve()
    if not lens_path.is_dir():
        raise NotADirectoryError(f"Lens directory not found: {lens_path}")

    os.chdir(lens_path)

    # pipeline.py lives in the same directory as this shared entry point.
    pipe_dir = Path(__file__).resolve().parent
    pipe_dir_str = str(pipe_dir)
    if pipe_dir_str not in sys.path:
        sys.path.insert(0, pipe_dir_str)

    from pipeline import main as pipeline_main  # noqa: E402

    initial_source = None
    if args.initial_source_from is not None:
        with args.initial_source_from.open("rb") as stream:
            initial_source = pickle.load(stream)["extra"]["s1"]
    pipeline_main(
        skip_done=args.skip_done, out_dir=args.out_dir,
        source_interpolation=args.source_interpolation, initial_source=initial_source,
    )


if __name__ == "__main__":
    main()
