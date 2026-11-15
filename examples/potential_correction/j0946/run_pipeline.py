import argparse
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
if __package__ in (None, ""):
    sys.path.insert(0, str(HERE.parents[2]))

from examples.potential_correction.j0946._handoff import load_handoff


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Run the J0946 macro fit and selected potential-correction "
        "branches in isolated processes."
    )
    parser.add_argument("--data-dir", type=Path, default=HERE / "data")
    parser.add_argument("--output-root", type=Path, default=HERE / "output")
    parser.add_argument(
        "--macro-dir",
        type=Path,
        help="Reuse a completed, verified M3 handoff instead of fitting the macro",
    )
    parser.add_argument(
        "--correction",
        choices=("single-step", "lbfgs", "both"),
        default="both",
    )
    parser.add_argument("--resume-macro", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument(
        "--source-interpolation", choices=("bilinear", "cubic_hermite")
    )
    args = parser.parse_args(argv)
    if args.macro_dir is not None and (
        args.resume_macro
        or args.prepare_only
        or args.source_interpolation is not None
    ):
        parser.error(
            "--macro-dir cannot be combined with --resume-macro, "
            "--prepare-only or --source-interpolation"
        )
    data = args.data_dir.expanduser().resolve()
    root = args.output_root.expanduser().resolve()
    macro = (
        args.macro_dir.expanduser().resolve()
        if args.macro_dir is not None
        else root / "macro"
    )
    branches = (
        ("single_step", "lbfgs")
        if args.correction == "both"
        else (args.correction.replace("-", "_"),)
    )
    outputs = [] if args.prepare_only else [root / branch for branch in branches]
    for destination in outputs + (
        [] if args.macro_dir is not None else [macro]
    ):
        if (
            destination == data
            or data in destination.parents
            or destination in data.parents
        ):
            parser.error("Outputs must be separate from the raw data directory")
    for destination in outputs:
        if (
            destination == macro
            or macro in destination.parents
            or destination in macro.parents
        ):
            parser.error(
                "Correction outputs must be separate from the M3 handoff"
            )
        if destination.exists():
            parser.error(
                f"Choose a new correction output directory: {destination}"
            )
    if args.macro_dir is None:
        missing = [
            name
            for name in ("image.fits", "noise.fits", "psf.fits")
            if not (data / name).is_file()
        ]
        if missing:
            parser.error(
                f"Missing raw FITS in {data}: {missing}; provide --data-dir"
            )
        if (
            macro.exists()
            and (not macro.is_dir() or any(macro.iterdir()))
            and not args.resume_macro
        ):
            parser.error(
                "Existing macro output requires --resume-macro or a new "
                "--output-root"
            )
        command = [
            sys.executable,
            str(HERE / "macro_model.py"),
            "--data-dir",
            str(data),
            "--output",
            str(macro),
        ]
        if args.resume_macro:
            command.append("--resume")
        if args.source_interpolation:
            command.extend(
                ["--source-interpolation", args.source_interpolation]
            )
        if args.prepare_only:
            command.append("--prepare-only")
        subprocess.run(command, check=True)
        if args.prepare_only:
            return
    load_handoff(macro)
    scripts = {"single_step": "pt.py", "lbfgs": "pt_iter_lbfgs.py"}
    for branch in branches:
        subprocess.run(
            [
                sys.executable,
                str(HERE / scripts[branch]),
                "--macro-dir",
                str(macro),
                "--output",
                str(root / branch),
            ],
            check=True,
        )


if __name__ == "__main__":
    main()
