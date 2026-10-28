"""Generate the matched Pseudo-Jaffe mock/control pair without fitting."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path

if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from examples.potential_correction.mock import MockConfig, generate_mock_pair, save_mock_pair


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path,
                        help="JSON input, or a previously saved configuration.json")
    parser.add_argument("--output", type=Path, default=Path("output/potential_correction_mock"))
    for name, kind in (("npix", int), ("dpix", float), ("nsub", int), ("seed", int),
                       ("psf_size", int), ("b_sub", float), ("theta_E", float),
                       ("subhalo_x", float), ("subhalo_y", float), ("noise_sigma", float)):
        parser.add_argument("--" + name.replace("_", "-"), dest=name, type=kind, default=None)
    parser.add_argument("--dtype", choices=("float32", "float64"), default=None)
    args = parser.parse_args(argv)
    values = asdict(MockConfig())
    if args.config is not None:
        supplied = json.loads(args.config.read_text())
        values.update(supplied.get("configuration", supplied))
    values.update({name: value for name, value in vars(args).items()
                   if name not in ("config", "output") and value is not None})
    pair = generate_mock_pair(MockConfig.from_dict(values))
    output = save_mock_pair(pair, args.output)
    print(f"Saved matched subhalo/control observations to {output.resolve()}")
    return output


if __name__ == "__main__":
    main()
