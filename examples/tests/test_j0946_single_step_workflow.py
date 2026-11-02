"""Numerical regressions for the self-contained J0946 M3 -> correction handoff."""

import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest


@pytest.fixture
def driver():
    path = Path(__file__).resolve().parents[2] / "workspace/single_step/pt.py"
    spec = importlib.util.spec_from_file_location("j0946_single_step_pt", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _handoff(path):
    path.mkdir()
    source = np.array([1.0, 1.2, 0.8, 1.1], dtype=np.float32)
    np.savez_compressed(
        path / "m3_handoff.npz",
        image_data=np.ones((4, 4), dtype=np.float32),
        noise_map=np.full((4, 4), 0.2, dtype=np.float32),
        psf_kernel=np.ones((1, 1), dtype=np.float32),
        lens_subtracted_image=np.ones((4, 4), dtype=np.float32),
        mask=np.zeros((4, 4), dtype=bool),
        source_values=source,
        source_bbox=np.array([-0.5, 0.5, -0.5, 0.5], dtype=np.float32),
        source_scale=np.array([1.0, 2.0, 4.0, 8.0], dtype=np.float32),
        macro_model_image=np.ones((4, 4), dtype=np.float32),
    )
    record = dict(
        complete=True,
        handoff_sha256=hashlib.sha256(
            (path / "m3_handoff.npz").read_bytes()
        ).hexdigest(),
        representative="test",
        macro_parameters=dict(
            theta_E=0.1,
            gamma=2.0,
            e1_mass=0.05,
            e2_mass=0.02,
            center_x_mass=0.0,
            center_y_mass=0.0,
            gamma1=0.01,
            gamma2=0.0,
        ),
        dpix=0.1,
        nsub=1,
        source_n=2,
        source_regularization="first-order",
        lambda_source=0.1,
    )
    (path / "m3_handoff.json").write_text(json.dumps(record))


@pytest.mark.parametrize(
    ("formulation", "fallback"), [("data", False), ("data", True), ("joint", False)]
)
def test_m3_handoff_dense_evidence_and_export_preserve_fixed_adaptation(
    driver, tmp_path, monkeypatch, formulation, fallback
):
    path = tmp_path / "macro"
    _handoff(path)
    with jax.experimental.enable_x64(False):
        problem, metadata, context = driver.load_problem(path)
        gx, gy = problem.renderer.source_spatial_gradient(
            jnp.asarray(context["source_values"]), problem.renderer.source_mapping()
        )
        np.testing.assert_array_equal(problem.gradient_x, gx)
        np.testing.assert_array_equal(problem.gradient_y, gy)
        evidence = driver.DenseEvidence(problem, formulation=formulation)
        for point in ([-1.0, -2.0, 1.0], [0.0, -1.0, 2.0]):
            metrics, values = evidence.solve(jnp.asarray(point, dtype=jnp.float32))
            ls, l4, l0 = 10.0 ** np.asarray(point)
            problem.lambda_amplitude = float(l0)
            reference = problem.fit(float(ls), float(l4))
            assert reference.success
            np.testing.assert_allclose(metrics[0], reference.log_evidence, atol=2e-4)
            np.testing.assert_allclose(
                values[: problem.n_source], reference.source_values, atol=2e-5
            )
            np.testing.assert_allclose(
                values[problem.n_source :], reference.potential_values, atol=2e-5
            )
            assert metrics.dtype == np.dtype("float32")
        if fallback:
            original = evidence.primary_solve

            def failed_primary(x):
                metrics, values = original(x)
                return metrics.at[0].set(-jnp.inf), values

            evidence.primary_solve = failed_primary
            metrics, values = evidence.solve(jnp.asarray(point, dtype=jnp.float32))
            assert np.isfinite(float(metrics[0])) and float(metrics[6]) == 1.0
            np.testing.assert_allclose(
                values[: problem.n_source], reference.source_values, atol=2e-5
            )
        result = driver.export_fit(problem, evidence, point, tmp_path / "fit", context)
        assert result.success
        actual_formulation = "joint" if fallback else formulation
        expected_method = (
            "dense_data_covariance_cholesky"
            if actual_formulation == "data"
            else "prior_whitened_augmented_qr"
        )
        expected_determinant = (
            "full_data_covariance_cholesky"
            if actual_formulation == "data"
            else "full_gaussian_augmented_qr"
        )
        assert result.solver_info.method == expected_method
        assert result.solver_info.iterations == (2 if fallback else 1)
        assert result.solver_info.diagnostics["fallback_float32"] is fallback
        assert result.determinant_method == expected_determinant
        assert result.configuration["dense_formulation"] == actual_formulation
        assert result.configuration["requested_dense_formulation"] == formulation
        assert result.configuration["fallback_float32"] is fallback
        saved = json.loads((tmp_path / "fit/result.json").read_text())
        assert saved["solver_info"]["method"] == expected_method
        assert saved["solver_info"]["diagnostics"]["fallback_float32"] is fallback
        assert saved["determinant_method"] == expected_determinant
        assert saved["configuration"]["dense_formulation"] == actual_formulation
        with np.load(tmp_path / "fit/reconstruction.npz") as output:
            np.testing.assert_array_equal(
                output["source_scale"], context["source_scale"]
            )
            assert output["potential_values"].dtype == np.dtype("float32")
            assert output["source_values"].dtype == np.dtype("float32")
        assert (
            metadata["handoff_sha256"]
            == result.initialization_provenance["handoff_sha256"]
        )
        (tmp_path / "manifest.json").write_text(json.dumps({"fixed_macro": metadata}))
        from matplotlib.axes import Axes

        images = []
        original_imshow = Axes.imshow

        def capture_imshow(ax, *args, **kwargs):
            artist = original_imshow(ax, *args, **kwargs)
            images.append(artist)
            return artist

        monkeypatch.setattr(Axes, "imshow", capture_imshow)
        driver.plot_diagnostics(result, context, tmp_path / "fit")
        assert (tmp_path / "fit/field_diagnostics.png").stat().st_size > 0
        assert (tmp_path / "fit/image_diagnostics.pdf").stat().st_size > 0
        source_images = [
            im
            for im in images
            if im.axes.get_title()
            in ("M3 source (gradient input)", "Jointly reconstructed source")
        ]
        assert len(source_images) == 2
        for im in source_images:
            left, right, bottom, top = im.get_extent()
            ny, nx = im.get_array().shape
            np.testing.assert_allclose(
                left + (np.arange(nx) + 0.5) * (right - left) / nx,
                result.source_x_axis,
            )
            np.testing.assert_allclose(
                bottom + (np.arange(ny) + 0.5) * (top - bottom) / ny,
                result.source_y_axis,
            )


@pytest.mark.parametrize("change", [None, "source_regularization", "macro_parameters"])
def test_resume_validates_json_handoff_before_dense_preparation(
    driver, tmp_path, monkeypatch, change
):
    path, output = tmp_path / "macro", tmp_path / "potential"
    _handoff(path)
    argv = [
        "pt.py",
        "--macro-dir",
        str(path),
        "--output",
        str(output),
        "--validate-only",
        "--log10-lambda-s",
        "-1",
        "-0.9",
        "--log10-lambda4",
        "-2",
        "-1.9",
        "--log10-lambda0",
        "1",
        "1.1",
    ]
    with jax.experimental.enable_x64(False):
        monkeypatch.setattr(sys, "argv", argv)
        driver.main()
        before = {p.name: p.read_bytes() for p in output.iterdir()}
        metadata_path = path / "m3_handoff.json"
        metadata = json.loads(metadata_path.read_text())
        if change == "source_regularization":
            metadata["source_regularization"] = "zero-order"
        elif change == "macro_parameters":
            metadata["macro_parameters"]["gamma1"] += 0.02
        metadata_path.write_text(json.dumps(metadata, indent=2))
        assert (
            metadata["handoff_sha256"]
            == hashlib.sha256((path / "m3_handoff.npz").read_bytes()).hexdigest()
        )
        monkeypatch.setattr(sys, "argv", argv + ["--resume"])
        if change is None:
            driver.main()  # Identical metadata remains compatible, even if reformatted.
        else:

            def unexpected_preparation(*args, **kwargs):
                pytest.fail("Changed handoff must be rejected before dense preparation")

            monkeypatch.setattr(driver, "DenseEvidence", unexpected_preparation)
            with pytest.raises(RuntimeError, match="fixed macro handoff differs"):
                driver.main()
            assert {p.name: p.read_bytes() for p in output.iterdir()} == before


def test_changed_handoff_is_rejected(driver, tmp_path):
    path = tmp_path / "macro"
    _handoff(path)
    with (path / "m3_handoff.npz").open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(RuntimeError, match="changed M3 handoff"):
        driver.load_problem(path)


@pytest.mark.parametrize(
    "change", [None, "arrays", "replacement", "metadata", "incomplete"]
)
def test_plot_only_validates_handoff_before_overwriting_figures(
    driver, tmp_path, monkeypatch, change
):
    macro, output = tmp_path / "macro", tmp_path / "potential"
    _handoff(macro)
    output.mkdir()
    metadata_path = macro / "m3_handoff.json"
    metadata = json.loads(metadata_path.read_text())
    (output / "manifest.json").write_text(
        json.dumps(dict(fixed_macro=metadata, settings=dict(potential_spacing=0.1)))
    )
    for name in ("maximum_evidence", "posterior_median"):
        folder = output / name
        folder.mkdir()
        np.savez_compressed(
            folder / "reconstruction.npz",
            source_x_axis=[-0.5, 0.5],
            source_y_axis=[-0.5, 0.5],
            potential_node_indices=[[0, 0]],
        )
        (folder / "field_diagnostics.png").write_bytes(b"original figure")
    np.savez_compressed(
        output / "posterior.npz",
        log10_strengths=np.array([[-1.0, -2.0, 1.0], [-0.9, -1.9, 1.1]]),
        weights=[0.5, 0.5],
    )
    before = {
        p.relative_to(output): p.read_bytes() for p in output.rglob("*") if p.is_file()
    }
    if change in ("arrays", "replacement"):
        path = macro / "m3_handoff.npz"
        with np.load(path) as archive:
            arrays = dict(archive)
        arrays["source_values"] = np.full(4, 99.0, dtype=np.float32)
        np.savez_compressed(path, **arrays)
        if change == "replacement":
            metadata["handoff_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    elif change == "metadata":
        metadata["macro_parameters"]["gamma1"] += 0.02
    elif change == "incomplete":
        metadata["complete"] = False
    metadata_path.write_text(json.dumps(metadata, indent=2))

    plotted = []

    def plot_fit(result, context, folder):
        np.testing.assert_array_equal(
            context["source_values"], np.float32([1, 1.2, 0.8, 1.1])
        )
        (folder / "field_diagnostics.png").write_bytes(b"redrawn figure")
        plotted.append(folder.name)

    def unexpected_solve(*args, **kwargs):
        pytest.fail("Plot-only must not prepare a solve")

    monkeypatch.setattr(driver, "plot_diagnostics", plot_fit)
    monkeypatch.setattr(
        driver, "plot_posterior", lambda *args: plotted.append("posterior")
    )
    monkeypatch.setattr(driver, "load_problem", unexpected_solve)
    monkeypatch.setattr(driver, "DenseEvidence", unexpected_solve)
    monkeypatch.setattr(
        sys,
        "argv",
        ["pt.py", "--macro-dir", str(macro), "--output", str(output), "--plot-only"],
    )
    if change is None:
        driver.main()
        assert plotted == ["maximum_evidence", "posterior_median", "posterior"]
    else:
        message = (
            "changed M3 handoff"
            if change in ("arrays", "incomplete")
            else "fixed macro handoff differs"
        )
        with pytest.raises(RuntimeError, match=message):
            driver.main()
        assert not plotted
        assert {
            p.relative_to(output): p.read_bytes()
            for p in output.rglob("*")
            if p.is_file()
        } == before


@pytest.mark.parametrize("symlink", [False, True])
def test_qr_audit_rejects_original_run_as_output_before_preparation(
    tmp_path, monkeypatch, symlink
):
    path = (
        Path(__file__).resolve().parents[2] / "workspace/single_step/audit_joint_qr.py"
    )
    spec = importlib.util.spec_from_file_location("j0946_qr_auditor", path)
    auditor = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(auditor)
    run = tmp_path / "original-run"
    run.mkdir()
    for name in ("manifest.json", "posterior.npz", "progress.json"):
        (run / name).write_bytes(b"original artifact")
    reconstruction = run / "maximum_evidence/reconstruction.npz"
    reconstruction.parent.mkdir()
    reconstruction.write_bytes(b"original reconstruction")
    output = run
    if symlink:
        output = tmp_path / "run-alias"
        output.symlink_to(run, target_is_directory=True)
    before = {p.relative_to(run): p.read_bytes() for p in run.rglob("*") if p.is_file()}

    def unexpected_preparation():
        pytest.fail("Output alias must be rejected before loading the numerical driver")

    monkeypatch.setattr(auditor, "load_driver", unexpected_preparation)
    with pytest.raises(ValueError, match="must differ from the original run"):
        auditor.evaluate(
            SimpleNamespace(
                run=run, output=output, macro=tmp_path / "macro", resume=False
            )
        )
    assert {
        p.relative_to(run): p.read_bytes() for p in run.rglob("*") if p.is_file()
    } == before


def test_resume_rejects_pre_qr_scores_before_dense_preparation(
    driver, tmp_path, monkeypatch
):
    path, output = tmp_path / "macro", tmp_path / "potential"
    _handoff(path)
    output.mkdir()
    (output / "manifest.json").write_text(
        json.dumps(
            dict(
                fixed_macro=json.loads((path / "m3_handoff.json").read_text()),
                settings=dict(
                    numerical_policy="float32_data_with_float32_joint_fallback_and_prior_QR"
                ),
            )
        )
    )
    (output / "checkpoint.h5").write_bytes(b"original-sampler-state")
    before = {p.name: p.read_bytes() for p in output.iterdir()}

    def unexpected_preparation(*args, **kwargs):
        pytest.fail("Legacy scores must be rejected before dense preparation")

    monkeypatch.setattr(driver, "DenseEvidence", unexpected_preparation)
    monkeypatch.setattr(
        sys,
        "argv",
        ["pt.py", "--macro-dir", str(path), "--output", str(output), "--resume"],
    )
    with pytest.raises(RuntimeError, match="numerical policy differs"):
        driver.main()
    assert {p.name: p.read_bytes() for p in output.iterdir()} == before


def test_float32_qr_prior_root_preserves_amplitude_in_fourth_difference_null_modes(
    driver,
):
    with jax.experimental.enable_x64(False):
        difference = jnp.array([[1.0, -4.0, 6.0, -4.0, 1.0]], dtype=jnp.float32)
        roughness = difference.T @ difference
        _, diagonal, factor = driver.potential_factor(
            roughness, difference, jnp.float32(1e6), jnp.float32(10.0)
        )
        root = (
            np.asarray(factor, dtype=float) * np.asarray(diagonal, dtype=float)[:, None]
        )
        # A constant potential lies in the D4 nullspace. Only lambda0 acts.
        np.testing.assert_allclose(
            np.linalg.norm(root.T @ np.ones(5)) ** 2, 50.0, rtol=2e-3
        )
        target = 1e6 * np.asarray(roughness, dtype=float) + 10.0 * np.eye(5)
        actual_logdet = 2 * np.log(abs(np.diag(root))).sum()
        np.testing.assert_allclose(
            actual_logdet, np.linalg.slogdet(target)[1], atol=2e-3
        )


@pytest.mark.parametrize("n", [3, 100])
@pytest.mark.parametrize("adaptive", [False, True], ids=["uniform", "adaptive"])
def test_macro_source_plot_centers_match_nodes_and_physical_overlays(
    tmp_path, monkeypatch, n, adaptive
):
    from TinyLensGpu.PhysicalModel import PhysicalModel, PixelizedSourceModel, Shear

    import caskade as ck
    import matplotlib.pyplot as plt

    bbox = (-0.6, 0.4, -0.4, 0.6)
    source_x = np.linspace(bbox[0], bbox[1], n)
    source_y = np.linspace(bbox[2], bbox[3], n)
    source = np.arange(n * n, dtype=float).reshape(n, n) / (n * n)
    scale = 1 + source if adaptive else None
    positions = np.array([[bbox[0], bbox[2]], [bbox[1], bbox[3]]])

    class PlotLikelihood(ck.Module):
        """Control inversion outputs; exercise the real plot and parameter context."""

        def __init__(self):
            super().__init__()
            pixelized = PixelizedSourceModel(
                n=n, log_lambda_reg=0.0, adaptive_reg_rho=1.5 if adaptive else 0.0
            )
            pixelized.log_lambda_reg.to_dynamic()
            self.phys_model = PhysicalModel(
                lens_mass=[Shear(gamma1=0.0, gamma2=0.0)],
                source_light=[pixelized],
                lens_light=[],
            )
            self.image_data = np.ones((25, 25))
            self.noise_map = np.full((25, 25), 0.1)
            self.unmask = np.ones((25, 25), dtype=bool)
            self._has_pos_penalty = True
            self._pos_px, self._pos_py = (jnp.asarray(axis) for axis in positions.T)

        def evaluate(self):
            return SimpleNamespace(
                model_image=np.full((25, 25), 0.9),
                source_pixels=source.ravel(),
                source_bbox=bbox,
            )

        def _get_reg_scale(self):
            return None if scale is None else scale.ravel()

        def _position_likelihood_penalty_jax(self):
            return jnp.asarray(0.0)

    caustic_x = np.array([bbox[0] - 0.2, bbox[1] + 0.2])
    caustic_y = np.array([0.1, 0.1])
    overlays = []

    def overlay_probe(*, image_axes, source_ax, lens_mass):
        source_ax.plot(caustic_x, caustic_y, label="Caustic probe")
        overlays.append(source_ax)

    # Load the actual plot functions without importing the CLI's runtime policy.
    path = Path(__file__).resolve().parents[2] / "workspace/single_step/macro_model.py"
    functions = [
        node
        for node in ast.parse(path.read_text(), filename=str(path)).body
        if isinstance(node, ast.FunctionDef)
        and node.name in {"_plot_pix_stage", "_source_param_value"}
    ]
    assert len(functions) == 2
    namespace = dict(
        np=np,
        jnp=jnp,
        plt=plt,
        ck=ck,
        DPIX=0.05,
        overlay_critical_and_caustics=overlay_probe,
    )
    exec(
        compile(ast.Module(body=functions, type_ignores=[]), str(path), "exec"),
        namespace,
    )
    figures = []
    original_subplots = plt.subplots

    def capture_subplots(*args, **kwargs):
        figure, axes = original_subplots(*args, **kwargs)
        figures.append((figure, axes))
        return figure, axes

    monkeypatch.setattr(plt, "subplots", capture_subplots)
    model = PlotLikelihood()
    artifact = tmp_path / "macro.png"
    namespace["_plot_pix_stage"](
        "stage-M3", model, {"log_lambda_reg": 0.2}, ["log_lambda_reg"], str(artifact)
    )
    assert artifact.stat().st_size > 0
    _, axes = figures[0]
    for ax, expected in ((axes[3], source), (axes[4], scale)):
        artist = ax.images[0]
        left, right, bottom, top = artist.get_extent()
        np.testing.assert_allclose(
            left + (np.arange(n) + 0.5) * (right - left) / n,
            source_x,
            rtol=0,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            bottom + (np.arange(n) + 0.5) * (top - bottom) / n,
            source_y,
            rtol=0,
            atol=1e-12,
        )
        np.testing.assert_array_equal(
            artist.get_array(), np.ones((n, n)) if expected is None else expected
        )
        np.testing.assert_allclose(ax.get_xlim(), [left, right])
        np.testing.assert_allclose(ax.get_ylim(), [bottom, top])
    assert overlays == [axes[3]]
    traced, caustic = axes[3].lines
    np.testing.assert_allclose(traced.get_xdata(), positions[:, 0], atol=1e-7)
    np.testing.assert_allclose(traced.get_ydata(), positions[:, 1], atol=1e-7)
    np.testing.assert_array_equal(caustic.get_xdata(), caustic_x)
    np.testing.assert_array_equal(caustic.get_ydata(), caustic_y)


def _five_node_gaussian_problem(coupled):
    """An exact Gaussian matrix problem, exercising the real cached fitter."""
    difference = jnp.array([[1.0, -4.0, 6.0, -4.0, 1.0]], dtype=jnp.float32)

    class SourcePrior:
        def matvec_free(self, source, *bbox, scale=None):
            return source

    class PotentialPrior:
        windows = jnp.arange(5, dtype=jnp.int32)[None, :]
        coefficients = difference[0]

        def to_dense(self, l4, l0):
            return l4 * (difference.T @ difference) + l0 * jnp.eye(5, dtype=jnp.float32)

        def forward(self, potential):
            return difference @ potential

        def matvec(self, potential, l4, l0):
            return l4 * difference.T @ (difference @ potential) + l0 * potential

    if coupled:
        design = np.array(
            [
                [1.0, 0.3, -0.1, 0.2, 0.4, -0.2],
                [0.2, -0.4, 0.2, 0.1, -0.3, 0.5],
                [-0.4, 0.1, 0.3, -0.2, 0.2, 0.1],
            ],
            dtype=np.float32,
        )
        data = np.array([1.0, -0.2, 0.5], dtype=np.float32)
        noise = np.array([0.7, 1.0, 0.8], dtype=np.float32)
    else:
        # Unobserved potential modes must integrate out to exactly one, even
        # when D4 makes their precision much stiffer than the amplitude prior.
        design = np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        data, noise = np.ones(1, dtype=np.float32), np.ones(1, dtype=np.float32)
    return SimpleNamespace(
        n_source=1,
        n_potential=5,
        dtype=jnp.dtype("float32"),
        dense_mapping=lambda: jnp.asarray(design),
        source_regularizer=SourcePrior(),
        potential_regularizer=PotentialPrior(),
        source_scale=None,
        renderer=SimpleNamespace(source_bbox=(-1.0, 1.0, -1.0, 1.0)),
        data_active=jnp.asarray(data),
        noise_active=jnp.asarray(noise),
        noise_logdet=2 * jnp.log(jnp.asarray(noise)).sum(),
        solve_rtol=1e-5,
    )


@pytest.mark.parametrize("fallback", [False, True], ids=["joint", "fallback"])
@pytest.mark.parametrize("coupled", [False, True], ids=["unobserved", "coupled"])
@pytest.mark.parametrize("lambda4", [1e6, 4e6])
def test_strong_d4_joint_and_fallback_complete_evidence(
    driver, monkeypatch, fallback, coupled, lambda4
):
    with jax.experimental.enable_x64(False):
        problem = _five_node_gaussian_problem(coupled)
        evidence = driver.DenseEvidence(
            problem, formulation="data" if fallback else "joint"
        )
        if fallback:
            primary = evidence.primary_solve

            def failed_primary(point):
                metrics, values = primary(point)
                return metrics.at[0].set(-jnp.inf), values

            monkeypatch.setattr(evidence, "primary_solve", failed_primary)
        point = jnp.asarray(np.log10([1e-3, lambda4, 10.0]), dtype=jnp.float32)
        metrics, values = evidence.solve(point)

        # Evaluate the analytic rank-one inverse in independent NumPy Float64,
        # at the actual strengths produced by the Float32 log10 transform.
        ls, l4, l0 = np.asarray(jnp.power(jnp.float32(10), point), dtype=float)
        d4 = np.array([1.0, -4.0, 6.0, -4.0, 1.0])
        inverse = np.zeros((6, 6))
        inverse[0, 0] = 1 / ls
        inverse[1:, 1:] = np.eye(5) / l0 - (
            l4 / (l0 * (l0 + l4 * (d4 @ d4)))
        ) * np.outer(d4, d4)
        design = np.asarray(evidence.design, dtype=float)
        data = np.asarray(problem.data_active, dtype=float)
        noise = np.asarray(problem.noise_active, dtype=float)
        covariance = np.diag(noise**2) + design @ inverse @ design.T
        dual = np.linalg.solve(covariance, data)
        expected_values = inverse @ design.T @ dual
        expected_logz = -0.5 * (
            data @ dual
            + np.linalg.slogdet(covariance)[1]
            + data.size * np.log(2 * np.pi)
        )
        expected_prior_logdet = (
            np.log(ls) + 4 * np.log(l0) + np.log(l0 + l4 * (d4 @ d4))
        )
        expected_posterior_logdet = (
            expected_prior_logdet
            + np.linalg.slogdet(covariance)[1]
            - 2 * np.log(noise).sum()
        )
        assert bool(metrics[6]) is fallback
        assert metrics.dtype == values.dtype == jnp.dtype("float32")
        # A stiff coupled solution can hit Float32's physical-residual floor.
        # Keep the original gate: validate the full ungated evidence as well
        # as the score returned when that independent certificate passes.
        ungated_logz = -0.5 * (
            float(metrics[2])
            + float(metrics[3])
            + float(metrics[5])
            - float(metrics[4])
            + float(problem.noise_logdet)
            + data.size * np.log(2 * np.pi)
        )
        np.testing.assert_allclose(ungated_logz, expected_logz, rtol=0, atol=1e-3)
        assert np.isfinite(float(metrics[0])) == (
            float(metrics[1]) <= problem.solve_rtol
        )
        if not coupled:
            assert np.isfinite(float(metrics[0]))
        if np.isfinite(float(metrics[0])):
            np.testing.assert_allclose(metrics[0], expected_logz, rtol=0, atol=1e-3)
        np.testing.assert_allclose(metrics[4], expected_prior_logdet, rtol=0, atol=1e-3)
        np.testing.assert_allclose(
            metrics[5], expected_posterior_logdet, rtol=0, atol=1e-3
        )
        np.testing.assert_allclose(values, expected_values, rtol=2e-3, atol=2e-5)
