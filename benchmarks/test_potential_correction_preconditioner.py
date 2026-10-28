"""Performance assertions stay separate from numerical core/example tests."""

import numpy as np
import pytest


@pytest.mark.performance
def test_cached_principal_factors_reduce_repeated_gpu_application_time():
    import jax

    if jax.devices()[0].platform != "gpu":
        pytest.skip("GPU block-application benchmark")
    from potential_correction_preconditioner import main

    record = main([])
    medians = {
        method: np.median(
            [
                row["seconds_per_apply"]
                for row in record["measurements"]
                if row["method"] == method
            ]
        )
        for method in ("triangular", "inverse_factor_matvec")
    }
    assert record["relative_l2_difference"] < 1e-12
    assert medians["inverse_factor_matvec"] < 0.5 * medians["triangular"]
