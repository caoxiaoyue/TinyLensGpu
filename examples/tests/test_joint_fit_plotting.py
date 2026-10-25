"""Regression checks for joint-fit example plotting."""

import ast

import pytest


_JOINT_FIT_FILES = (
    "pix_src_demo/simple/pix_src_lens_light/fit_joint.py",
    "pix_src_demo/simple/pix_src_lens_light_mge/fit_joint.py",
)


@pytest.mark.unit
@pytest.mark.parametrize("fit_path", _JOINT_FIT_FILES, ids=["fit_path0", "fit_path1"])
def test_joint_fit_overlay_uses_correct_axes_and_parameter_context(fit_path, examples_dir):
    """The critical-curve overlay uses 2D axes indexing and the fitted context."""
    import TinyLensGpu  # noqa: F401 -- import first so caskade picks 'jax' backend
    import caskade as ck
    import numpy as np

    fit_path = examples_dir / fit_path
    tree = ast.parse(fit_path.read_text(), filename=str(fit_path))
    overlay_stmt = None
    for node in tree.body:
        for sub in ast.walk(node):
            if (
                isinstance(sub, ast.Call)
                and isinstance(sub.func, ast.Name)
                and sub.func.id == "overlay_critical_and_caustics"
            ):
                overlay_stmt = node
                break
        if overlay_stmt is not None:
            break
    assert overlay_stmt is not None, f"overlay call not found in {fit_path}"

    calls = []
    axes = np.asarray(
        [[object() for _ in range(4)] for _ in range(2)], dtype=object,
    )
    prob_model = ck.Module("overlay_probe")
    prob_model.phys_model = ck.Module("mass_probe")
    prob_model.phys_model.theta_E = ck.Param("theta_E", 1.0)
    prob_model.phys_model.theta_E.to_dynamic()

    def overlay_critical_and_caustics(*, image_axes, source_ax, lens_mass):
        calls.append(dict(
            image_axes=image_axes,
            source_ax=source_ax,
            lens_mass=lens_mass,
            theta_E=float(np.asarray(lens_mass.theta_E.value)),
        ))

    namespace = {
        "np": np, "ck": ck,
        "axes": axes,
        "prob_model": prob_model,
        "best_params": [1.1],
        "overlay_critical_and_caustics": overlay_critical_and_caustics,
    }
    exec(
        compile(
            ast.Module(body=[overlay_stmt], type_ignores=[]),
            str(fit_path), "exec",
        ),
        namespace,
    )
    assert len(calls) == 1
    call = calls[0]
    assert call["image_axes"] == [axes[0, 0], axes[0, 1], axes[0, 2]]
    assert call["source_ax"] is axes[0, 3]
    assert call["lens_mass"] is prob_model.phys_model
    assert call["theta_E"] == pytest.approx(1.1)
