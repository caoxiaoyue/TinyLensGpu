import json
import time
from pathlib import Path

import jax
import numpy as np

from demo_pix_src import simulate_lensing_data, setup_pixelized_model


def _time_call(fn, n_runs: int) -> dict:
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        out = fn()
        if hasattr(out, "block_until_ready"):
            out.block_until_ready()
        elif isinstance(out, tuple):
            for x in out:
                if hasattr(x, "block_until_ready"):
                    x.block_until_ready()
                    break
        t1 = time.perf_counter()
        times.append(t1 - t0)

    arr = np.array(times, dtype=np.float64)
    return {
        "n_runs": int(n_runs),
        "mean_s": float(arr.mean()),
        "std_s": float(arr.std(ddof=1) if n_runs > 1 else 0.0),
        "min_s": float(arr.min()),
        "max_s": float(arr.max()),
        "all_s": [float(x) for x in arr.tolist()],
    }


def main(n_runs: int = 10, seed: int = 0) -> dict:
    np.random.seed(seed)

    data_dict = simulate_lensing_data()
    prob_model = setup_pixelized_model(data_dict)

    inverter, source_mesh_beta, blurred_lens_map_matrix = prob_model._get_or_build_inverter()

    _ = inverter.solve().block_until_ready()
    _ = inverter.log_evidence().block_until_ready()

    results = {
        "jax_backend": jax.default_backend(),
        "jax_devices": [str(d) for d in jax.devices()],
        "seed": int(seed),
        "npix": int(data_dict["noisy_image"].shape[0]),
        "dpix": float(data_dict["dpix"]),
        "n_data": int(np.sum(~data_dict["mask"])),
        "n_source": int(source_mesh_beta.shape[0]),
        "F_shape": [int(x) for x in blurred_lens_map_matrix.shape],
        "timing": {
            "solve": _time_call(lambda: inverter.solve(), n_runs=n_runs),
            "log_evidence": _time_call(lambda: inverter.log_evidence(), n_runs=n_runs),
        },
    }

    out_path = Path(__file__).with_name("bench_speed_pix_src_results.json")
    out_path.write_text(json.dumps(results, indent=2, sort_keys=True))
    return results


if __name__ == "__main__":
    res = main()
    print(json.dumps(res["timing"], indent=2, sort_keys=True))
