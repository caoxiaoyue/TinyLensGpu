import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from TinyLensGpu.ForwardSimulation import SimulatorConfig
from TinyLensGpu.PhysicalModel import PhysicalModel, SIS, Shear
from .model import (
    MockConfig,
    MockObservation,
    MockPair,
    SubhaloTruth,
    generate_mock_pair,
)

DEFAULT_DATA = Path(__file__).resolve().parent / "data" / "default"


def load_mock_pair(data):
    data = Path(data)
    metadata = json.loads((data / "configuration.json").read_text())
    if (
        metadata.get("integration", {}).get("masked_input_light")
        != "full cutout evaluated before PSF"
    ):
        raise ValueError(
            "Dataset uses an unsupported PSF/mask convention; regenerate it"
        )
    config = MockConfig.from_dict(metadata["configuration"])
    observations = []
    with jax.experimental.enable_x64(config.dtype == "float64"):
        scalar = lambda value: jnp.asarray(value, dtype=config.dtype)
        macro = PhysicalModel(
            lens_mass=[
                SIS(
                    theta_E=scalar(config.theta_E),
                    center_x=scalar(config.macro_center_x),
                    center_y=scalar(config.macro_center_y),
                ),
                Shear(gamma1=scalar(config.gamma1), gamma2=scalar(config.gamma2)),
            ],
            source_light=[], lens_light=[],
        )
        for case, strength in (("subhalo", config.b_sub), ("control", 0.0)):
            with np.load(data / f"{case}.npz", allow_pickle=False) as stored:
                arrays = {
                    name: stored[name]
                    for name in (
                        "image_data",
                        "noise_map",
                        "noise_realization",
                        "model_image",
                        "psf_kernel",
                        "mask",
                    )
                }
            shape = (config.npix, config.npix)
            for name in (
                "image_data",
                "noise_map",
                "noise_realization",
                "model_image",
                "mask",
            ):
                if arrays[name].shape != shape:
                    raise ValueError(
                        f"{case}/{name} does not match dataset image shape"
                    )
            if arrays["mask"].dtype != np.dtype(bool) or arrays["mask"].all():
                raise ValueError(
                    f"{case}: mask must select pixels and have boolean dtype"
                )
            for name in (
                "image_data",
                "noise_map",
                "noise_realization",
                "model_image",
                "psf_kernel",
            ):
                if arrays[name].dtype != np.dtype(config.dtype) or not np.isfinite(
                    arrays[name]
                ).all():
                    raise ValueError(
                        f"{case}/{name} must have finite {config.dtype} values"
                    )
            if (arrays["noise_map"] <= 0).any():
                raise ValueError(f"{case}: noise must be positive")
            if arrays["psf_kernel"].shape != (
                config.psf_size,
                config.psf_size,
            ) or not np.isclose(arrays["psf_kernel"].sum(), 1.0):
                raise ValueError(f"{case}: invalid PSF shape or normalization")
            values = {name: jnp.asarray(value) for name, value in arrays.items()}
            sim = SimulatorConfig(
                config.dpix,
                config.npix,
                values["psf_kernel"],
                config.nsub,
                values["mask"],
            )
            observations.append(
                MockObservation(
                    **values,
                    sim_config=sim,
                    macro=macro,
                    source_truth=config.source,
                    subhalo_truth=SubhaloTruth(
                        strength,
                        config.theta_E,
                        config.subhalo_x,
                        config.subhalo_y,
                    ),
                    config=config,
                    has_subhalo=strength > 0,
                )
            )
    pair = MockPair(*observations)
    for name in ("noise_map", "noise_realization", "mask", "psf_kernel"):
        if not np.array_equal(
            getattr(pair.subhalo, name), getattr(pair.control, name)
        ):
            raise ValueError(f"Matched observations must share {name}")
    return pair


def prepare_mock_input(configuration, *, data=None, output):
    config = dict(configuration)
    recorded_dataset = config.get("dataset") if data is None else None
    if recorded_dataset is not None:
        data = recorded_dataset["path"]
    if data is None and "mock" in config:
        return config, generate_mock_pair(MockConfig.from_dict(config["mock"]))
    data = Path(DEFAULT_DATA if data is None else data).resolve()
    output = Path(output).resolve()
    if output == data or data in output.parents or output in data.parents:
        raise ValueError(
            "Output must be separate from the input dataset directory"
        )
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Choose a new output directory: {output}")
    hashes = {
        name: hashlib.sha256((data / name).read_bytes()).hexdigest()
        for name in ("configuration.json", "subhalo.npz", "control.npz")
    }
    if recorded_dataset is not None and hashes != recorded_dataset["sha256"]:
        raise ValueError(
            "Saved dataset checksums changed; refusing to reuse this run configuration"
        )
    pair = load_mock_pair(data)
    config["mock"] = asdict(pair.subhalo.config)
    config["dataset"] = {"path": str(data), "sha256": hashes}
    return config, pair
