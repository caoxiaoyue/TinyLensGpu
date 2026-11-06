"""Device-resident component probes; run AFTER end-to-end timing workers.

Times include Python dispatch and synchronization, not pure CUDA kernel time.
The explicit inverse-whitening matrix is an exploratory microbenchmark only.
"""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import statistics
from time import perf_counter

from run_gauss_newton import (
    ROOT,
    jax,
    jnp,
    jsl,
    np,
    GaussNewtonPotentialCorrection,
    MockConfig,
    generate_mock_pair,
    SimulatorConfig,
    PotentialCorrectionRenderer,
    IterativePotentialCorrection,
    write_json,
    threadpool_limits,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("subhalo", "control"), default="subhalo")
    parser.add_argument("--endpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--calls", type=int, default=40)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "results.json").exists():
        raise RuntimeError("Use a new output directory")
    shutil.copyfile(Path(__file__), args.output / "microbench_snapshot.py")
    with jax.experimental.enable_x64(), jax.default_matmul_precision(
        "highest"
    ), threadpool_limits(limits=16):
        config = json.loads(
            (
                ROOT
                / "examples/potential_correction/iterative_scientific_configuration.json"
            ).read_text()
        )
        mock = MockConfig.from_dict(config["mock"])
        observation = getattr(generate_mock_pair(mock), args.case)
        simulator = SimulatorConfig(
            mock.dpix,
            mock.npix,
            observation.psf_kernel,
            config["inversion_nsub"],
            observation.mask,
        )
        renderer = PotentialCorrectionRenderer(
            observation.macro,
            simulator,
            config["source_n"],
            tuple(config["source_bbox"]),
            potential_spacing=config["potential_spacing"],
            dtype=mock.dtype,
        )
        sx, sy = jnp.meshgrid(renderer.source_x_axis, renderer.source_y_axis)
        source = observation.source_truth.brightness(sx, sy).ravel()
        problem = IterativePotentialCorrection(
            renderer,
            observation.image_data,
            observation.noise_map,
            source,
            lambda_amplitude=config["lambda_amplitude"],
            source_regularization=config["source_regularization"],
            solve_rtol=config["solve_rtol"],
            solve_atol=config["solve_atol"],
        )
        strength = config["fixed_strengths"][args.case]
        strengths = (strength["lambda_source"], strength["lambda_potential"])
        prepared = GaussNewtonPotentialCorrection(problem, strengths, 5e-4)
        n_data = int(problem.data_active.size)
        metric = jnp.asarray(prepared.matrix(problem.initial_values))
        chol = jnp.linalg.cholesky(metric)
        prior = jnp.linalg.cholesky(problem.dense_prior(*strengths))
        with np.load(args.endpoint, allow_pickle=False) as saved:
            z = jnp.asarray(saved["z"])
            saved_values = np.array(saved["values"])
        point = prepared.decode(z, chol)
        np.testing.assert_allclose(point, saved_values, rtol=1e-7, atol=1e-9)
        rng = np.random.default_rng(42)
        vector = jnp.asarray(rng.normal(size=problem.n_joint))
        vector /= jnp.linalg.norm(vector)
        weights = jnp.asarray(rng.normal(size=n_data + problem.n_joint))
        weights /= jnp.linalg.norm(weights)
        image_weights = weights[:n_data]
        linear = prepared.linearization_factory(z, chol, prior)
        jax.block_until_ready(linear)
        solve = jax.jit(lambda c, v: jsl.solve_triangular(c.T, v, lower=False))
        solve_transpose = jax.jit(lambda c, v: jsl.solve_triangular(c, v, lower=True))
        gemv = jax.jit(lambda a, v: a @ v)
        prior_push = jax.jit(lambda a, v: a.T @ v)
        weighted_image = lambda x: problem.forward(x) / problem.noise_active
        image_push = jax.jit(lambda x, v: jax.jvp(weighted_image, (x,), (v,))[1])
        image_pull = jax.jit(lambda x, u: jax.vjp(weighted_image, x)[1](u)[0])
        started = perf_counter()
        transform = jsl.solve_triangular(
            chol.T, jnp.eye(problem.n_joint, dtype=problem.dtype), lower=False
        )
        transform.block_until_ready()
        transform_seconds = perf_counter() - started
        relative_inverse_residual = float(
            jnp.linalg.norm(chol.T @ transform - jnp.eye(problem.n_joint))
            / jnp.sqrt(problem.n_joint)
        )
        transform_error = float(
            jnp.linalg.norm(transform @ vector - solve(chol, vector))
            / jnp.linalg.norm(solve(chol, vector))
        )
        np.testing.assert_allclose(
            transform @ vector, solve(chol, vector), rtol=1e-7, atol=1e-10
        )

        def dense_residual(q, mapping, precision_factor):
            values = problem.initial_values + mapping @ q
            return jnp.concatenate(
                (
                    (problem.forward(values) - problem.data_active)
                    / problem.noise_active,
                    precision_factor.T @ values,
                )
            )

        dense_factory = jax.jit(
            lambda q, m, l: jax.linearize(lambda a: dense_residual(a, m, l), q)[1]
        )
        dense_linear = dense_factory(z, transform, prior)
        jax.block_until_ready(dense_linear)
        np.testing.assert_allclose(
            prepared.cached_push(dense_linear, vector),
            prepared.push(z, chol, prior, vector),
            rtol=1e-7,
            atol=1e-8,
        )
        np.testing.assert_allclose(
            prepared.cached_pull(dense_linear, weights),
            prepared.pull(z, chol, prior, weights),
            rtol=1e-7,
            atol=1e-8,
        )
        functions = {
            "whitening_triangular": lambda: solve(chol, vector),
            "whitening_triangular_transpose": lambda: solve_transpose(chol, vector),
            "whitening_explicit_gemv": lambda: gemv(transform, vector),
            "prior_cholesky_gemv": lambda: prior_push(prior, vector),
            "physical_image_jvp": lambda: image_push(point, vector),
            "physical_image_vjp": lambda: image_pull(point, image_weights),
            "full_direct_jvp": lambda: prepared.push(z, chol, prior, vector),
            "full_direct_vjp": lambda: prepared.pull(z, chol, prior, weights),
            "full_cached_jvp": lambda: prepared.cached_push(linear, vector),
            "full_cached_vjp": lambda: prepared.cached_pull(linear, weights),
            "full_cached_dense_whitening_jvp": lambda: prepared.cached_push(
                dense_linear, vector
            ),
            "full_cached_dense_whitening_vjp": lambda: prepared.cached_pull(
                dense_linear, weights
            ),
        }
        for function in functions.values():
            function().block_until_ready()
        samples = {name: [] for name in functions}
        for repeat in range(args.repeats):
            for name in rng.permutation(list(functions)):
                started = perf_counter()
                for _ in range(args.calls):
                    functions[name]().block_until_ready()
                samples[name].append((perf_counter() - started) / args.calls)
        results = dict(
            case=args.case,
            n_joint=problem.n_joint,
            n_data=n_data,
            jax=jax.__version__,
            gpu=[d.device_kind for d in jax.devices()],
            endpoint=str(args.endpoint),
            endpoint_sha256=hashlib.sha256(args.endpoint.read_bytes()).hexdigest(),
            runner_sha256=hashlib.sha256(
                Path(__file__).with_name("run_gauss_newton.py").read_bytes()
            ).hexdigest(),
            calls=args.calls,
            repeats=args.repeats,
            units="seconds per synchronized device-resident call; includes dispatch/wait",
            transform_build_seconds=transform_seconds,
            transform_bytes=int(transform.size * transform.dtype.itemsize),
            relative_inverse_residual=relative_inverse_residual,
            relative_transform_vector_error=transform_error,
            samples=samples,
            median={name: statistics.median(times) for name, times in samples.items()},
            caution="Component times are diagnostic, not additive independent kernels. Explicit-transform speed is not yet end-to-end validated.",
        )
        write_json(args.output / "results.json", results)
        for name, value in results["median"].items():
            print(name, f"{value*1000:.5f} ms", flush=True)


if __name__ == "__main__":
    main()
