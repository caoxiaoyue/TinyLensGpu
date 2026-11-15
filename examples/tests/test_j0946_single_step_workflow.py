"""Numerical regressions for the self-contained J0946 M3 -> correction handoff."""

import ast
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest


J0946_EXAMPLE = (
    Path(__file__).resolve().parents[1] / "potential_correction" / "j0946"
)


@pytest.fixture
def driver():
    path = J0946_EXAMPLE / "pt.py"
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
        selected_sample_index=0,
    )
    (path / "m3_handoff.json").write_text(json.dumps(record))


@pytest.mark.parametrize("consumer", ["single_step", "lbfgs"])
@pytest.mark.parametrize("interpolation", [None, "bilinear", "cubic_hermite"])
def test_j0946_consumers_use_the_handoff_interpolation(
    tmp_path, driver, consumer, interpolation
):
    from TinyLensGpu.example_support import source_interpolation_metadata
    from examples.potential_correction.j0946 import pt_iter_lbfgs

    handoff = tmp_path / "macro"
    _handoff(handoff)
    metadata_file = handoff / "m3_handoff.json"
    metadata = json.loads(metadata_file.read_text())
    if interpolation is not None:
        metadata.update(source_interpolation_metadata(interpolation))
        metadata_file.write_text(json.dumps(metadata))
    mode = interpolation or "bilinear"
    with jax.experimental.enable_x64(consumer == "lbfgs"):
        if consumer == "single_step":
            problem, saved, arrays = driver.load_problem(handoff)
        else:
            problem, saved, arrays = pt_iter_lbfgs.prepare_problem(handoff, 0.2)
    assert saved == metadata == json.loads(metadata_file.read_text())
    fitted_mode = mode if consumer == "single_step" else "cubic_hermite"
    assert problem.renderer.source_interpolation == fitted_mode
    assert (
        problem.renderer.source_boundary
        == source_interpolation_metadata(fitted_mode)["source_boundary"]
    )
    with jax.experimental.enable_x64(consumer == "lbfgs"):
        assert problem.renderer.source_mapping().weights.shape[1] == (
            16 if fitted_mode == "cubic_hermite" else 4
        )
    assert arrays["source_values"].shape == (4,)
    np.testing.assert_array_equal(problem.source_scale, arrays["source_scale"])
    provenance = problem.initialization_provenance
    assert provenance["handoff_sha256"] == metadata["handoff_sha256"]
    if consumer == "single_step":
        assert (
            provenance["adaptive_scale"]
            == "fixed_M1_template_and_M2_rho_as_used_in_M3"
        )
        assert provenance["gradient"] == f"frozen_{mode}_derivative_of_M3_source"
        with jax.experimental.enable_x64(False):
            gx, gy = problem.renderer.source_spatial_gradient(
                arrays["source_values"]
            )
        np.testing.assert_array_equal(problem.gradient_x, gx)
        np.testing.assert_array_equal(problem.gradient_y, gy)
    else:
        assert problem.reference_interpolation == mode
        assert provenance["whitening_source_interpolation"] == mode
        assert provenance["m3_sample_index"] == 0
        np.testing.assert_array_equal(
            problem.initial_values[:4], arrays["source_values"]
        )
        np.testing.assert_array_equal(
            problem.reference_values[:4], arrays["source_values"]
        )
        np.testing.assert_array_equal(problem.initial_values[4:], 0.0)


@pytest.fixture
def macro_producer(monkeypatch):
    monkeypatch.setattr(sys, "path", sys.path.copy())
    path = J0946_EXAMPLE / "macro_model.py"
    spec = importlib.util.spec_from_file_location("j0946_macro_test", path)
    module = importlib.util.module_from_spec(spec)
    precision = jax.config.jax_default_matmul_precision
    try:
        spec.loader.exec_module(module)
        yield module
    finally:
        jax.config.update("jax_default_matmul_precision", precision)


@pytest.mark.parametrize("interpolation", [None, "cubic_hermite"])
def test_macro_resume_preserves_the_recorded_source_interpolation(
    tmp_path, monkeypatch, macro_producer, interpolation
):
    from TinyLensGpu.example_support import source_interpolation_metadata

    raw = tmp_path / "raw"
    raw.mkdir()
    for name in ("image.fits", "noise.fits", "psf.fits"):
        (raw / name).write_bytes(name.encode())
    output = tmp_path / "macro"
    monkeypatch.setattr(macro_producer, "DATA_DIR", tmp_path / "data")
    options = [
        "macro_model.py", "--prepare-only", "--out-dir", str(output),
        "--input-dir", str(raw),
    ]
    requested = [] if interpolation is None else ["--source-interpolation", interpolation]
    monkeypatch.setattr(sys, "argv", options + requested)
    macro_producer.cli()
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    mode = interpolation or "bilinear"
    expected = source_interpolation_metadata(mode)
    for key, value in expected.items():
        assert manifest["settings"][key] == value
        if interpolation is None:
            # Simulate a known pre-feature manifest without changing its inputs.
            del manifest["settings"][key]
    manifest_path.write_text(json.dumps(manifest))
    preserved = manifest_path.read_bytes()

    monkeypatch.setattr(sys, "argv", options + ["--resume"])
    macro_producer.cli()
    assert macro_producer.SOURCE_INTERPOLATION == mode
    assert manifest_path.read_bytes() == preserved

    other = "cubic_hermite" if mode == "bilinear" else "bilinear"
    monkeypatch.setattr(sys, "argv", options + ["--resume", "--source-interpolation", other])
    with pytest.raises(ValueError, match="Start a fresh run"):
        macro_producer.cli()

    monkeypatch.setattr(sys, "argv", options + ["--resume"])
    (output / "sampler_m1.h5").touch()
    (output / "sampler_m1.json").write_text(json.dumps(source_interpolation_metadata(other)))
    with pytest.raises(ValueError, match="sampler_m1.json"):
        macro_producer.cli()
    assert manifest_path.read_bytes() == preserved


@pytest.fixture
def small_macro_fit(macro_producer, tmp_path, monkeypatch):
    from TinyLensGpu.Inference import StagePosterior

    for name, value in (
        ("NSRC", 3), ("NSUB_PIX", 1), ("OUT_DIR", tmp_path),
        ("SOURCE_INTERPOLATION", "cubic_hermite"),
    ):
        monkeypatch.setattr(macro_producer, name, value)
    macro = dict(
        theta_E=0.12, e1_mass=0.04, e2_mass=0.02,
        center_x_mass=0.01, center_y_mass=-0.02, gamma1=0.01, gamma2=0.02,
        center_x_src=0.0, center_y_src=0.0,
    )
    center = np.array(list(macro.values()))
    stage_a = StagePosterior.from_schema(
        np.stack([center - 0.001, center, center + 0.001]),
        [0.25, 0.5, 0.25], param_names=list(macro), log_z=0.0,
    )
    image = np.full((7, 7), 0.2, dtype=np.float32)
    noise = np.full_like(image, 0.1)
    psf = np.ones((1, 1), dtype=np.float32)
    mask = np.zeros_like(image, dtype=bool)
    return image, noise, psf, mask, stage_a


def test_macro_exports_the_native_reference_and_selected_m3_fields(
    macro_producer, small_macro_fit, driver, tmp_path, monkeypatch,
):
    import caskade as ck
    from TinyLensGpu.Inference import StagePosterior

    image, noise, psf, mask, stage_a = small_macro_fit
    args = image, noise, psf, mask, stage_a, None
    log_lambda = float(np.log(0.1))
    with jax.experimental.enable_x64(False):
        m0 = macro_producer.build_stage_m0_likelihood(*args)
        assert m0.source_model.interpolation == "cubic_hermite"
        m1 = macro_producer.build_stage_m1_likelihood(*args, log_lambda)
        m1_params = {
            param.name: float(param.value) for param in m1.get_dynamic_params()
        }
        s1 = macro_producer._solve_pixel_source_for_package(
            m1, m1_params, list(m1_params),
        )
        assert s1["source_interpolation"] == "cubic_hermite"
        with ck.ActiveContext(m1):
            m1.fill_params(jnp.asarray(list(m1_params.values())))
            reference, _, _ = m1.reconstruct_component_images(
                jnp.asarray(s1["source_pixels"], dtype=m1.image_data.dtype), None,
            )
        assert s1["source_image"].shape == (3, 3)
        np.testing.assert_allclose(s1["adapt_image"], reference, atol=1e-7, rtol=1e-6)
        s1.update(lambda_best=0.1, log_lambda_best=log_lambda)
        m2 = macro_producer.build_stage_m2_likelihood(
            image, noise, psf, mask, m1_params, None, s1,
        )
        assert m2.source_model.interpolation == "cubic_hermite"
        regs = dict(log_lambda_reg=log_lambda, adaptive_reg_rho=0.5)
        m3 = macro_producer.build_stage_m3_likelihood(
            image, noise, psf, mask, stage_a, None, regs, s1,
        )
        assert m3.source_model.interpolation == "cubic_hermite"
        m3_params = {
            param.name: float(param.value) for param in m3.get_dynamic_params()
        }
        names = list(m3_params)
        samples = np.array([list(m3_params.values())] * 2)
        samples[1, names.index("theta_E")] += 0.05
        evaluated = [m3.evaluate_with_diagnostics(jnp.asarray(draw)) for draw in samples]
        scores = np.array([float(result.log_evidence) for result, _ in evaluated])
        selected = int(np.argmax(scores))
        assert evaluated[selected][1].valid and evaluated[selected][1].evidence_valid
        assert scores[0] != scores[1]
        stage_m3 = StagePosterior.from_schema(
            samples, [0.5, 0.5], param_names=names, log_z=0.0,
        )
        np.savez(
            tmp_path / "posterior_m3.npz", samples=samples, log_likelihood=scores,
            param_names=names,
        )
        calls = []
        evaluate = type(m3).evaluate_with_diagnostics

        def record_evaluation(model, point):
            result, diagnostics = evaluate(model, point)
            calls.append((np.asarray(point), result, diagnostics))
            return result, diagnostics

        monkeypatch.setattr(type(m3), "evaluate_with_diagnostics", record_evaluation)
        macro_producer.export_m3_bundle(
            image, image, noise, psf, mask, np.zeros_like(image), stage_a, stage_m3,
            None, regs, s1, mask,
        )
    assert len(calls) == 1
    selected_point, result, diagnostics = calls[0]
    assert diagnostics.valid and diagnostics.evidence_valid
    np.testing.assert_array_equal(selected_point, samples[selected].astype(np.float32))
    with jax.experimental.enable_x64(False):
        problem, record, arrays = driver.load_problem(tmp_path)
    assert record["source_interpolation"] == "cubic_hermite"
    assert record["gradient_provenance"] == "cubic_hermite_derivative_of_new_M3_reconstruction"
    assert record["evidence_logdet"] == diagnostics.evidence_logdet == "overlap"
    assert record["objective"]["reference_id"]
    assert record["objective"]["adaptation"]["recipe"] == "native_source_arc_v1"
    assert record["source_scale_provenance"] == "selected_M3_native_reference_projection"
    assert problem.initialization_provenance["adaptive_scale"] == record["source_scale_provenance"]
    assert record["selected_sample_index"] == selected
    assert record["best_sample_log_likelihood"] == float(result.log_evidence)
    assert record["retained_sample_log_likelihood"] == scores[selected]
    assert problem.renderer.source_boundary == record["source_boundary"]
    np.testing.assert_array_equal(arrays["source_bbox"], s1["source_bbox"])
    np.testing.assert_array_equal(arrays["source_values"], result.source_pixels)
    from examples.potential_correction.j0946.pt_iter_lbfgs import prepare_problem

    with jax.experimental.enable_x64():
        nonlinear, nonlinear_record, nonlinear_arrays = prepare_problem(
            tmp_path, 0.2
        )
    assert nonlinear_record == record
    assert nonlinear.reference_interpolation == record["source_interpolation"]
    assert nonlinear.initialization_provenance["m3_sample_index"] == selected
    np.testing.assert_array_equal(
        nonlinear.source_scale, arrays["source_scale"]
    )
    np.testing.assert_array_equal(
        nonlinear.initial_values[: nonlinear.n_source], arrays["source_values"]
    )
    np.testing.assert_array_equal(
        nonlinear_arrays["source_values"], arrays["source_values"]
    )
    np.testing.assert_array_equal(arrays["macro_model_image"], result.model_image)
    np.testing.assert_array_equal(
        arrays["source_scale"], np.asarray(diagnostics.regularization_scale).ravel(),
    )
    np.testing.assert_array_equal(problem.source_scale, arrays["source_scale"])
    assert arrays["source_values"].dtype == np.float32


def test_macro_resume_rejects_an_incompatible_scored_checkpoint(
    macro_producer, small_macro_fit, tmp_path, monkeypatch,
):
    image, noise, psf, mask, stage_a = small_macro_fit
    runs = []

    class ControlledSampler:
        def __init__(self, prior, loglike, *, n_dim, filepath, **options):
            self.points = np.asarray([prior(np.full(n_dim, q)) for q in (0.49, 0.5, 0.51)])
            self.scores = np.asarray(loglike(self.points))
            self.n_like = self.n_eff = len(self.points)
            self.log_z = float(np.max(self.scores))
            filepath.touch()
            runs.append(n_dim)

        def run(self, **options):
            return True

        def posterior(self):
            return self.points, np.log([0.2, 0.6, 0.2]), self.scores

    monkeypatch.setattr(macro_producer, "Sampler", ControlledSampler)
    monkeypatch.setattr(macro_producer, "load_lens_data", lambda **kw: (image, noise, psf, None))
    monkeypatch.setattr(macro_producer, "run_stage_b", lambda *a, **kw: mask)
    monkeypatch.setattr(macro_producer, "_position_likelihood_from_stage_a", lambda point: None)
    monkeypatch.setattr(macro_producer, "_plot_pix_stage", lambda *a: None)
    macro_producer._dump_stage(
        "a", stage_a.samples, stage_a.weights, stage_a.param_names, stage_a.log_z,
        extra=dict(medians=stage_a.medians(), lens_light_model=np.zeros_like(image)),
        stage=stage_a,
    )

    options = dict(skip_done=True, source_interpolation="cubic_hermite")
    with jax.experimental.enable_x64(False):
        macro_producer.main(**options)
        assert runs == [8, 2, 8]
        objectives = {
            tag: macro_producer._load_stage(tag)["extra"]["objective"]
            for tag in ("m0", "m1", "m2", "m3")
        }
        assert {objective["evidence_logdet"] for objective in objectives.values()} == {"overlap"}
        assert objectives["m2"]["reference_id"] == objectives["m3"]["reference_id"]
        for tag in ("m1", "m2", "m3"):
            checkpoint = json.loads((tmp_path / f"sampler_{tag}.json").read_text())
            assert checkpoint["objective"] == objectives[tag]["fingerprint"]

        handoff = tmp_path / "m3_handoff.npz"
        original = handoff.read_bytes()
        macro_producer.main(**options)
        assert runs == [8, 2, 8]
        assert handoff.read_bytes() == original

        # Simulate a checkpoint that still uses an older adaptive objective.
        path = tmp_path / "sampler_m2.json"
        checkpoint = json.loads(path.read_text())
        checkpoint.pop("objective")
        path.write_text(json.dumps(checkpoint))
        with pytest.raises(RuntimeError, match="Incompatible existing sampler"):
            macro_producer.main(**options)
        assert runs == [8, 2, 8]
        assert handoff.read_bytes() == original


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
    path = J0946_EXAMPLE / "macro_model.py"
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
