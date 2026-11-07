"""Source compatibility checks for runnable examples and documentation snippets."""

import ast
import importlib
import re

import pytest


REMOVED_OPERATOR_API = {
    "JointOperatorData",
    "_A_jit_prebound",
    "_joint_A_matvec_jit",
    "as_solver_protocol",
    "build_A_matvec",
    "call_A_matvec",
}


def _example_code_units(examples_dir):
    test_dir = examples_dir / "tests"
    for path in sorted(examples_dir.rglob("*.py")):
        if test_dir in path.parents:
            continue
        yield path.relative_to(examples_dir).as_posix(), path.read_bytes()

    python_fence = re.compile(r"```python\s*\n(.*?)```", re.DOTALL)
    for path in sorted(examples_dir.rglob("*.md")):
        if test_dir in path.parents:
            continue
        relative_path = path.relative_to(examples_dir).as_posix()
        for block_index, match in enumerate(
            python_fence.finditer(path.read_text()), start=1
        ):
            yield f"{relative_path}#python-{block_index}", match.group(1)


@pytest.mark.unit
def test_example_python_sources_compile(examples_dir):
    """Every Python source and Markdown Python block should compile."""
    for label, source in _example_code_units(examples_dir):
        compile(source, label, "exec")


@pytest.mark.unit
def test_example_tinylens_imports_resolve(examples_dir):
    """Names imported from TinyLensGpu should exist on their stated surface."""
    failures = []
    for label, source in _example_code_units(examples_dir):
        tree = ast.parse(source, filename=label)
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom):
                continue
            if node.module is None or not node.module.startswith("TinyLensGpu"):
                continue
            module = importlib.import_module(node.module)
            for alias in node.names:
                if alias.name != "*" and not hasattr(module, alias.name):
                    failures.append(
                        f"{label}:{node.lineno}: "
                        f"{node.module}.{alias.name}"
                    )
    assert not failures, "Invalid TinyLensGpu example imports:\n" + "\n".join(
        failures
    )


@pytest.mark.unit
def test_examples_do_not_use_removed_operator_api(examples_dir):
    """Examples should use the typed curvature-operator API exclusively."""
    failures = []
    for label, source in _example_code_units(examples_dir):
        tree = ast.parse(source, filename=label)
        used_names = {
            node.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Name)
        }
        used_names.update(
            node.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Attribute)
        )
        removed_names = sorted(used_names & REMOVED_OPERATOR_API)
        if removed_names:
            failures.append(f"{label}: {', '.join(removed_names)}")
    assert not failures, "Examples use removed operator API:\n" + "\n".join(
        failures
    )


_SURVEY_DIAGNOSTIC_SCRIPTS = (
    "slacs_examples/pipeline.py",
    "euclid_examples/pipeline.py",
    "pre_subtract_lens/pix_src_pipe/model.py",
    "pre_subtract_lens/pix_src_pipe_sersic/model.py",
    "pre_subtract_lens/pix_src_pipe_mge/model.py",
    "pre_subtract_lens/pix_src_pipe_mge/model_split.py",
)


def _load_survey_callers(examples_dir, relative_path):
    """Load existing script functions without data-loading or sampler side effects."""
    import TinyLensGpu  # noqa: F401 -- initialize the Caskade backend first
    import caskade as ck
    import jax.numpy as jnp
    import matplotlib.pyplot as plt
    import numpy as np

    from TinyLensGpu.ObservationModel.LensImage.inversion_result import (
        InvalidReconstructionError,
    )
    from TinyLensGpu.visualizer import overlay_critical_and_caustics

    path = examples_dir / "pix_src_demo_operator" / "pipe" / relative_path
    tree = ast.parse(path.read_text(), filename=str(path))
    selected = [
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in {
            "_plot_pix_stage", "_unmasked_extent",
            "_source_axes_from_bbox", "_solve_pixel_source_for_package",
        }
    ]
    from TinyLensGpu.example_support import (
        export_source_package, validate_source_package,
        require_valid_reconstruction, source_axes_from_bbox,
        draw_image_panel,
    )

    namespace = dict(
        np=np, jnp=jnp, ck=ck, plt=plt, DPIX=0.08, NSRC=3,
        InvalidReconstructionError=InvalidReconstructionError,
        overlay_critical_and_caustics=overlay_critical_and_caustics,
        export_source_package=export_source_package,
        validate_source_package=validate_source_package,
        require_valid_reconstruction=require_valid_reconstruction,
        source_axes_from_bbox=source_axes_from_bbox,
        draw_image_panel=draw_image_panel,
    )
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"),
         namespace)
    return namespace


@pytest.fixture
def survey_source_model():
    """Small real source-only operator with PSF, mask and position constraints."""
    import TinyLensGpu  # noqa: F401
    import jax
    import jax.numpy as jnp
    import numpy as np

    from TinyLensGpu.ForwardSimulation.LensImage.config import SimulatorConfig
    from TinyLensGpu.ForwardSimulation.LensImage.pixelized import PixelizedLensSimulator
    from TinyLensGpu.Inference import ParamU
    from TinyLensGpu.ObservationModel.LensImage.pixelized_image_model_operator import (
        PixelizedImageProbModelOperator,
    )
    from TinyLensGpu.PhysicalModel import PhysicalModel, PixelizedSourceModel, SIE

    mass = SIE(theta_E=0.12, e1=0.05, e2=0.0, center_x=0.0, center_y=0.0)
    for param in (mass.theta_E, mass.e1, mass.e2, mass.center_x, mass.center_y):
        param.to_static()
    mass.theta_E.to_dynamic()
    source = PixelizedSourceModel(
        n=3, log_lambda_reg=0.0, regularization_type="first-order",
        adaptive_reg_rho=ParamU("adaptive_reg_rho", 0.6),
    )
    source.log_lambda_reg.to_dynamic()
    source.adaptive_reg_rho.to_dynamic()
    phys = PhysicalModel(lens_mass=[mass], source_light=[source], lens_light=[])
    psf = jnp.asarray([[0.0, 0.1, 0.0], [0.1, 0.6, 0.1], [0.0, 0.1, 0.0]])
    bbox = (-0.4, 0.4, -0.4, 0.4)
    sim = PixelizedLensSimulator(
        phys, SimulatorConfig(dpix=0.08, npix=10, psf_kernel=psf),
    )
    image = sim.simulate(jnp.linspace(0.1, 1.0, 9), source_bbox=bbox)
    mask = np.ones((10, 10), dtype=bool)
    mask[2:8, 1:9] = False
    # Compare repeated float32 GPU solves at the same matmul precision.
    with jax.default_matmul_precision("highest"):
        model = PixelizedImageProbModelOperator(
            image, jnp.ones_like(image) * 0.05, psf, 0.08, phys,
            mask=mask, solver_type="pcg", block_size=2, fixed_source_bbox=bbox,
            fixed_reg_template=jnp.abs(jnp.linspace(-1.0, 1.0, 9)),
            position_likelihood=dict(
                positions=[[0.2, 0.0], [-0.2, 0.0]],
                threshold_arcsec=0.01, min_log_like=-20.0,
            ),
        )
        # Tighten only this tiny fixture so independent public evaluations can
        # be compared below the production solver's stopping tolerance.
        model.pcg_rtol = 1e-7
        yield model


@pytest.mark.integration
@pytest.mark.parametrize("relative_path", _SURVEY_DIAGNOSTIC_SCRIPTS)
@pytest.mark.parametrize("rho", [0.0, 1.2])
def test_survey_plot_renders_public_reconstruction(
    relative_path, rho, examples_dir, survey_source_model, tmp_path, monkeypatch,
    capsys,
):
    """A survey diagnostic renders the one public reconstruction at median theta."""
    import jax.numpy as jnp
    import matplotlib.pyplot as plt
    import numpy as np

    model = survey_source_model
    names = [param.name for param in model.get_dynamic_params()]
    medians = dict(theta_E=0.15, log_lambda_reg=0.2, adaptive_reg_rho=rho)
    theta = jnp.asarray([medians[name] for name in names])
    initial = np.asarray(model.get_values("flat")).copy()
    expected, diagnostics = model.evaluate_with_diagnostics(
        theta, include_effective_dof=True,
    )
    assert diagnostics.valid
    namespace = _load_survey_callers(examples_dir, relative_path)
    calls = []
    evaluate = model.evaluate_with_diagnostics

    def evaluate_once(*args, **kwargs):
        calls.append(kwargs)
        return evaluate(*args, **kwargs)

    monkeypatch.setattr(model, "evaluate_with_diagnostics", evaluate_once)
    figures = []
    savefig = plt.savefig

    def capture_figure(*args, **kwargs):
        figures.append(plt.gcf())
        return savefig(*args, **kwargs)

    monkeypatch.setattr(plt, "savefig", capture_figure)
    overlay = namespace["overlay_critical_and_caustics"]
    overlay_parameters = []

    def capture_overlay(**kwargs):
        overlay_parameters.append(np.asarray(model.get_values("flat")).copy())
        return overlay(**kwargs)

    namespace["overlay_critical_and_caustics"] = capture_overlay
    path = tmp_path / "survey.png"
    namespace["_plot_pix_stage"]("survey", model, medians, names, path)
    assert calls == [{"include_effective_dof": True}]
    assert path.read_bytes().startswith(b"\x89PNG")
    axes = [axis for axis in figures[0].axes if axis.images]
    split = relative_path.endswith("model_split.py")
    euclid = relative_path.startswith("euclid")
    assert len(axes) == (4 if split else 5)
    active = np.asarray(model.unmask)
    np.testing.assert_allclose(
        axes[1].images[0].get_array()[active],
        np.asarray(expected.model_image)[active], rtol=2e-5, atol=2e-5,
    )
    np.testing.assert_allclose(
        axes[2].images[0].get_array()[active],
        np.asarray(diagnostics.normalized_residuals)[active], rtol=2e-5, atol=2e-5,
    )
    np.testing.assert_allclose(
        axes[3].images[0].get_array(), np.asarray(expected.source_pixels).reshape(3, 3),
        rtol=2e-5, atol=2e-5,
    )
    np.testing.assert_allclose(axes[3].images[0].get_extent(), expected.source_bbox)
    assert f"χ²/ν={diagnostics.reduced_chi2:.3f}" in axes[2].get_title()
    assert f"λ={diagnostics.lambda_reg:.2e}" in axes[3].get_title()
    assert [axis.images[0].get_cmap().name for axis in axes[:4]] == [
        "viridis", "viridis", "RdBu_r", "viridis",
    ]
    np.testing.assert_allclose(axes[2].images[0].get_clim(), (-3, 3))
    np.testing.assert_array_equal(
        np.ma.getmaskarray(axes[2].images[0].get_array()), ~active,
    )
    if euclid or split:
        for axis in axes[:2]:
            np.testing.assert_array_equal(
                np.ma.getmaskarray(axis.images[0].get_array()), ~active,
            )
    # Existing variants intentionally use different crops: center bounds for
    # Euclid, pixel edges for split MGE, three-pixel padding for the other cases.
    xlim = (-0.28, 0.28) if euclid else (-0.32, 0.32) if split else (-0.4, 0.4)
    ylim = (-0.2, 0.2) if euclid else (-0.24, 0.24) if split else (-0.4, 0.4)
    for axis in axes[:3]:
        np.testing.assert_allclose(axis.get_xlim(), xlim, atol=1e-7)
        np.testing.assert_allclose(axis.get_ylim(), ylim, atol=1e-7)
    if not split:
        np.testing.assert_allclose(
            axes[4].images[0].get_array(), diagnostics.regularization_scale,
        )
        assert axes[4].images[0].get_cmap().name == "plasma"
        assert f"rho={diagnostics.adaptive_reg_rho:.2f}" in axes[4].get_title()
        for axis, label, coordinates in [
            (axes[0], "Marked pos", diagnostics.position.image_positions),
            (axes[3], "Traced pos", diagnostics.position.source_positions),
        ]:
            line = next(line for line in axis.lines if line.get_label() == label)
            np.testing.assert_allclose(
                np.asarray(line.get_data()).T, coordinates, rtol=2e-5, atol=2e-5,
            )
        output = capsys.readouterr().out
        assert f"{diagnostics.position.penalty:.4e}" in output
        assert f"{diagnostics.position.max_separation:.4e} arcsec" in output
    np.testing.assert_allclose(overlay_parameters, [theta])
    np.testing.assert_array_equal(model.get_values("flat"), initial)


def test_source_export_helper_preserves_legacy_default_schema(survey_source_model):
    from TinyLensGpu.example_support import export_source_package

    model = survey_source_model
    names = [param.name for param in model.get_dynamic_params()]
    medians = dict(theta_E=0.15, log_lambda_reg=0.2, adaptive_reg_rho=0.0)
    package = export_source_package(model, medians, names, n=3)
    assert set(package) == {
        "source_pixels", "source_image", "source_bbox", "source_x_axis", "source_y_axis", "n",
    }


@pytest.mark.integration
@pytest.mark.parametrize("relative_path", _SURVEY_DIAGNOSTIC_SCRIPTS[:-1])
@pytest.mark.parametrize("rho", [0.0, 1.2])
def test_survey_source_export_preserves_public_products(
    relative_path, rho, examples_dir, survey_source_model, tmp_path, monkeypatch,
):
    """Raw source packages retain float64 arrays, axes and keys without optional work."""
    import pickle

    import jax.numpy as jnp
    import numpy as np

    model = survey_source_model
    names = [param.name for param in model.get_dynamic_params()]
    medians = dict(theta_E=0.15, log_lambda_reg=0.2, adaptive_reg_rho=rho)
    theta = jnp.asarray([medians[name] for name in names])
    initial = np.asarray(model.get_values("flat")).copy()
    expected, diagnostics = model.evaluate_with_diagnostics(theta)
    assert diagnostics.valid
    namespace = _load_survey_callers(examples_dir, relative_path)
    evaluate = model.evaluate_with_diagnostics
    calls = []

    def evaluate_once(*args, **kwargs):
        calls.append(kwargs)
        return evaluate(*args, **kwargs)

    monkeypatch.setattr(model, "evaluate_with_diagnostics", evaluate_once)
    package = namespace["_solve_pixel_source_for_package"](model, medians, names)
    assert calls == [{}]
    path = tmp_path / "source.pkl"
    path.write_bytes(pickle.dumps(package))
    saved = pickle.loads(path.read_bytes())
    assert set(saved) == {
        "source_pixels", "source_image", "source_bbox",
        "source_x_axis", "source_y_axis", "n", "source_interpolation", "source_boundary",
    }
    assert saved["n"] == 3
    assert saved["source_interpolation"] == "bilinear"
    assert saved["source_boundary"] == "hard_zero_outside_bbox"
    for key in ("source_pixels", "source_image", "source_x_axis", "source_y_axis"):
        assert saved[key].dtype == np.float64
    np.testing.assert_allclose(
        saved["source_pixels"], expected.source_pixels, rtol=2e-5, atol=2e-5,
    )
    np.testing.assert_allclose(
        saved["source_image"], np.asarray(expected.source_pixels).reshape(3, 3),
        rtol=2e-5, atol=2e-5,
    )
    np.testing.assert_allclose(saved["source_bbox"], expected.source_bbox)
    np.testing.assert_allclose(saved["source_x_axis"], [-0.4, 0.0, 0.4])
    np.testing.assert_allclose(saved["source_y_axis"], [-0.4, 0.0, 0.4])
    np.testing.assert_array_equal(model.get_values("flat"), initial)


@pytest.mark.integration
@pytest.mark.parametrize("relative_path", _SURVEY_DIAGNOSTIC_SCRIPTS)
@pytest.mark.parametrize("failure", ["unconverged", "nonfinite"])
def test_survey_invalid_reconstruction_stops_output(
    relative_path, failure, examples_dir, survey_source_model, tmp_path, monkeypatch,
):
    """Real invalid solves raise before a figure/export, retaining earlier artifacts."""
    import jax.numpy as jnp
    import matplotlib.pyplot as plt
    import numpy as np

    from TinyLensGpu.ObservationModel.LensImage.pixelized_image_model_operator import (
        PixelizedImageProbModelOperator,
    )

    model = survey_source_model
    if failure == "unconverged":
        model.pcg_max_iter = 0
    else:
        model = PixelizedImageProbModelOperator(
            np.full((10, 10), np.nan), model.noise_map,
            jnp.asarray([[1.0]]), 0.08, model.phys_model,
            mask=~np.asarray(model.unmask), solver_type="pcg", block_size=2,
            fixed_source_bbox=(-0.4, 0.4, -0.4, 0.4),
            fixed_reg_template=jnp.abs(jnp.linspace(-1.0, 1.0, 9)),
        )
    names = [param.name for param in model.get_dynamic_params()]
    medians = dict(theta_E=0.15, log_lambda_reg=0.2, adaptive_reg_rho=0.0)
    theta = jnp.asarray([medians[name] for name in names])
    _, diagnostics = model.evaluate_with_diagnostics(theta)
    assert not diagnostics.valid
    assert diagnostics.invalid_reason
    namespace = _load_survey_callers(examples_dir, relative_path)
    evaluate = model.evaluate_with_diagnostics
    calls = []

    def evaluate_once(*args, **kwargs):
        calls.append(kwargs)
        return evaluate(*args, **kwargs)

    monkeypatch.setattr(model, "evaluate_with_diagnostics", evaluate_once)
    existing = tmp_path / "stage_previous.pkl"
    existing.write_bytes(b"previous stage")
    figures = plt.get_fignums()
    path = tmp_path / "invalid.png"
    with pytest.raises(RuntimeError, match="failed while plotting invalid") as error:
        namespace["_plot_pix_stage"]("invalid", model, medians, names, path)
    assert "converge" in str(error.value) or "nonfinite" in str(error.value)
    assert calls == [{"include_effective_dof": True}]
    assert plt.get_fignums() == figures
    assert not path.exists()
    if "_solve_pixel_source_for_package" in namespace:
        with pytest.raises(RuntimeError, match="failed"):
            namespace["_solve_pixel_source_for_package"](model, medians, names)
        assert calls == [{"include_effective_dof": True}, {}]
    assert list(tmp_path.iterdir()) == [existing]
    assert existing.read_bytes() == b"previous stage"


@pytest.mark.integration
@pytest.mark.parametrize("relative_path", _SURVEY_DIAGNOSTIC_SCRIPTS[:2])
def test_survey_stage_propagates_invalid_plot_and_retains_posterior(
    relative_path, examples_dir, survey_source_model, tmp_path,
):
    """A real stage must not swallow diagnostic failures after saving its posterior."""
    import pickle

    import jax.numpy as jnp
    import numpy as np

    path = examples_dir / "pix_src_demo_operator" / "pipe" / relative_path
    tree = ast.parse(path.read_text(), filename=str(path))
    # Execute existing definitions/imports/constants only, not entry-point
    # chdir, metadata reads, environment changes or JAX precision policy.
    selected = [
        node for node in tree.body
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef))
        or (
            isinstance(node, ast.Assign)
            and all(isinstance(target, ast.Name) and target.id.isupper()
                    for target in node.targets)
        )
    ]
    namespace = {}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"),
         namespace)
    namespace.update(NSRC=3, DPIX=0.08, NSUB_PIX=1, OUT_DIR=tmp_path)

    # The sampler is the external system boundary; builders, priors, posterior
    # transfer, stage persistence and numerical diagnostics remain real.
    class CompletedSampler:
        log_z = -12.0

        def __init__(self, prior, loglike, n_dim, **kwargs):
            assert n_dim == 2

        def run(self, **kwargs):
            pass

        def posterior(self):
            return np.array([[0.2, 1.2], [0.2, 1.2]]), np.log([0.5, 0.5]), None

    namespace["Sampler"] = CompletedSampler
    model = survey_source_model
    names = [param.name for param in model.get_dynamic_params()]
    package = namespace["_solve_pixel_source_for_package"](
        model, dict(theta_E=0.15, log_lambda_reg=0.2, adaptive_reg_rho=0.0), names,
    )
    package.update(lambda_best=float(np.exp(0.2)), log_lambda_best=0.2)
    mass_medians = dict(
        theta_E=0.15, gamma=2.0, e1_mass=0.05, e2_mass=0.0,
        center_x_mass=0.0, center_y_mass=0.0, gamma1=0.0, gamma2=0.0,
    )
    with pytest.raises(RuntimeError, match="failed while plotting stage-M2"):
        namespace["run_stage_m2"](
            np.full((10, 10), np.nan), model.noise_map, jnp.asarray([[1.0]]),
            ~np.asarray(model.unmask), np.zeros((10, 10)), mass_medians, None,
            package,
        )
    saved = pickle.loads((tmp_path / "stage_m2.pkl").read_bytes())
    assert saved["param_names"] == ["log_lambda_reg", "adaptive_reg_rho"]
    np.testing.assert_array_equal(saved["samples"], [[0.2, 1.2], [0.2, 1.2]])
    assert not (tmp_path / "stage_m2_model.png").exists()
