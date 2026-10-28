"""Replot saved successful public scientific results without refitting.

The original numerical result metadata and arrays remain byte-for-byte intact.
A separate record identifies the presentation code and source exports.
"""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import jax
from examples.potential_correction.mock import MockConfig, generate_mock_pair
from examples.potential_correction import result_products


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    record = json.loads((args.run / "run.json").read_text())
    mock = MockConfig.from_dict(record["configuration"]["mock"])
    manifest = {
        "scope": "Presentation-only regeneration from saved public results; no fit or numerical metadata rewrite",
        "git_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip(),
        "plot_module_sha256": hashlib.sha256(
            Path(result_products.__file__).read_bytes()
        ).hexdigest(),
        "sources": {},
    }
    with jax.experimental.enable_x64(mock.dtype == "float64"):
        pair = generate_mock_pair(mock)
        for key in record["runs"]:
            directory = args.run / key
            if (
                not (directory / "result.json").exists()
                or (directory / "failure.json").exists()
            ):
                continue
            hashes = {
                name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
                for name in ["result.json", "reconstruction.npz"]
            }
            result = result_products.load_result(directory)
            result_products.plot_result_products(
                result, getattr(pair, key.split("/")[0]), directory
            )
            assert hashes == {
                name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
                for name in hashes
            }
            manifest["sources"][key] = hashes
    result_products.write_json(args.run / "plot-provenance.json", manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
