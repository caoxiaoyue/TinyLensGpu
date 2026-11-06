"""Matched coarse-noise stress test: 2312 data, 100x100 source, fixed potential.

Extra independent noise makes the total coarse sigma exactly 0.01.
Two fixed prior profiles, three extra-noise seeds, and one cold plus three
warm fits each; raw failures are retained without optimizer fallback.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from time import perf_counter

from run_source_resolution import ROOT,HERE,CONFIG,OPTIONS,write,sha,runtime


def worker(args):
    from dataclasses import replace
    import gc
    from threadpoolctl import threadpool_limits
    from scipy.optimize import minimize
    import scipy
    jax, jnp, np, cfg, obs, SimulatorConfig, Renderer = runtime()
    import jax.scipy.linalg as jsl
    from TinyLensGpu.ObservationModel import IterativePotentialCorrection
    from run_scientific import digest
    from benchmark_scientific import sync
    from source_resolution_prior import PhysicalCurvaturePrior
    from run_experiment import objective_for
    from data_resolution_observation import ObservationProjection, ProjectedPotentialCorrection
    from coarse_noise import set_projected_likelihood

    with np.load(args.output/"frozen_inputs.npz") as archive:
        saved = {k:archive[k] for k in archive.files}
    frozen = json.loads((args.output/"frozen_inputs.json").read_text())
    assert all(digest(value) == frozen["hashes"][key] for key,value in saved.items())
    obs = replace(obs, **{k:jnp.asarray(saved[k]) for k in
        ("image_data", "noise_map", "psf_kernel", "mask", "model_image", "noise_realization")})
    sim = SimulatorConfig(cfg["mock"]["dpix"], cfg["mock"]["npix"], obs.psf_kernel,
                          cfg["inversion_nsub"], obs.mask)
    renderer = Renderer(obs.macro, sim, args.n, cfg["source_bbox"],
                        potential_spacing=cfg["potential_spacing"], dtype="float64")
    grid = renderer.potential_grid
    current_geometry = dict(potential_node_indices=grid.node_indices, potential_x=grid.x,
        potential_y=grid.y, potential_core_indices=grid.core_indices, query_x=renderer.query_x,
        query_y=renderer.query_y, macro_beta_x=renderer.macro_mapping.beta_x,
        macro_beta_y=renderer.macro_mapping.beta_y)
    assert all(digest(value) == frozen["hashes"][key] for key,value in current_geometry.items())
    protocol = json.loads((args.output/"protocol.json").read_text())
    strengths = protocol["profiles"][args.profile]
    ls, l4, l0 = [strengths[k] for k in ("lambda_source", "lambda_potential", "lambda_amplitude")]
    projection=ObservationProjection(saved["mask"],args.factor)
    coarse=projection
    noise_path=args.output/f"noise_seed_{args.seed:02d}.npz"
    expected_noise_hash=protocol["noise_inputs"][str(args.seed)]["sha256"]
    if sha(noise_path)!=expected_noise_hash:
        raise ValueError("Frozen noise input checksum mismatch")
    with np.load(noise_path) as archive:
        noisy={k:archive[k] for k in archive.files}
    problem = ProjectedPotentialCorrection(renderer, obs.image_data, obs.noise_map,
        jnp.zeros(renderer.n_source), lambda_amplitude=l0, source_regularization="second-order",
        solve_rtol=cfg["solve_rtol"], solve_atol=cfg["solve_atol"], projection=projection)
    set_projected_likelihood(problem,noisy["data"],noisy["sigma"])
    # Set all data/noise/prior arrays before tracing any kernels.
    problem.source_regularizer = PhysicalCurvaturePrior(args.n, reference_n=49)
    problem.initialization_provenance.update(source_truth_used=False,
        source_initialization="data_only_source_MAP_at_zero_potential",
        macro_fixed_to_truth=True, source_prior="physical_integrals_matched_to_n49")
    root = args.output/args.profile/f"seed_{args.seed:02d}"
    root.mkdir(parents=True)
    write(root/"environment.json", dict(python=platform.python_version(), jax=jax.__version__,
        numpy=np.__version__, scipy=scipy.__version__, devices=[d.device_kind for d in jax.devices()],
        dimensions=problem._configuration(), source_spacing_ratio=problem.source_regularizer.spacing_ratio,
        observation="binned",noise_seed=args.seed,noise_input_sha256=expected_noise_hash,
        target_sigma=protocol["target_sigma"],projection_shape=projection.shape,n_observations=projection.n_data,
        coarse_mask=projection.mask.tolist(),groups_disjoint=True,noise_min=float(problem.noise_active.min()),
        noise_max=float(problem.noise_active.max()),matched_fine_count=coarse.n_data*coarse.group_size,
        input_hashes=frozen["hashes"], checked_geometry_identical=True, strengths=strengths,
        source_initialization_reference_lambda=cfg["fixed_strengths"]["subhalo"]["lambda_source"]))

    def metric_builder(forward, prior, size):
        @jax.jit
        def block(point, indices):
            _, push = jax.linearize(forward, point)
            _, pull = jax.vjp(forward, point)
            return jax.vmap(lambda v: pull(push(v))[0]+prior(v))(
                jax.nn.one_hot(indices, size, dtype=point.dtype))

        def matrix(point):
            result = np.empty((size,size), dtype=np.float64)
            for start in range(0, size, args.panel):
                count = min(args.panel, size-start)
                values = np.asarray(block(point, jnp.arange(start, start+args.panel, dtype=jnp.int32)))
                result[:, start:start+count] = values[:count].T
            if not np.isfinite(result).all():
                raise FloatingPointError("Nonfinite AD curvature")
            return .5*(result+result.T)
        return matrix

    mapping = problem.macro_mapping
    source_forward = lambda s: problem.source_forward(s, renderer.source_mapping(macro_mapping=mapping))/problem.noise_active
    init_ls = cfg["fixed_strengths"]["subhalo"]["lambda_source"]
    source_prior = lambda s: init_ls*problem.source_regularizer.matvec_free(s, *renderer.source_bbox)
    source_metric = metric_builder(source_forward, source_prior, problem.n_source)
    _, pull = jax.vjp(source_forward, jnp.zeros(problem.n_source))
    rhs = pull(problem.data_active/problem.noise_active)[0]
    metric = metric_builder(lambda q:problem.forward(q)/problem.noise_active,
        lambda q:problem.prior_matvec(q, ls, l4), problem.n_joint)
    objective = jax.jit(objective_for(problem, (ls,l4)))
    gradient = jax.jit(jax.value_and_grad(objective))

    @jax.jit
    def factor(h):
        return jnp.linalg.cholesky(h)

    @jax.jit
    def decode(z, base, chol):
        return base+jsl.solve_triangular(chol.T, z, lower=False)

    white_vg = jax.jit(jax.value_and_grad(lambda z,base,chol:objective(decode(z,base,chol))))

    def checked_factor(h):
        chol = factor(jnp.asarray(h))
        sync(chol)
        if not bool(jnp.all(jnp.isfinite(chol)) & jnp.all(jnp.diag(chol)>0)):
            raise FloatingPointError("Float64 Cholesky failed; no jitter or fallback")
        return chol

    def directional_profile(q, g, chol):
        white_g = jsl.solve_triangular(chol, g, lower=True)
        norm = float(jnp.linalg.norm(white_g))
        if norm == 0:
            return dict(zero_gradient=True)
        direction = jsl.solve_triangular(chol.T, -white_g/norm, lower=False)
        def coordinates(point):
            rays = renderer.source_mapping(point[problem.n_source:])
            box = np.asarray(renderer.source_bbox)
            return np.column_stack(((np.asarray(rays.beta_x)-box[0])/(box[1]-box[0]),
                                    (np.asarray(rays.beta_y)-box[2])/(box[3]-box[2])))*(args.n-1)
        uv = coordinates(q)
        interior = np.all((uv>0)&(uv<args.n-1), axis=1)
        probes = []
        f0 = float(objective(q))
        for step in (1e-3, 1e-5, 1e-7, 1e-9):
            plus,minus = q+step*direction, q-step*direction
            a,b = coordinates(plus),coordinates(minus)
            probes.append(dict(step=step, objective_plus_minus_base=float(objective(plus))-f0,
                predicted_forward_change=-step*norm,
                interior_queries_changing_cells=int(np.count_nonzero(np.any(np.floor(a)!=np.floor(b), axis=1)&interior)),
                original_bbox_crossings=int(np.count_nonzero(np.any((a>=0)!=(b>=0),axis=1)|np.any((a<=args.n-1)!=(b<=args.n-1),axis=1)))))
        return dict(minimum_interior_knot_distance=float(np.abs(uv[interior]-np.round(uv[interior])).min()),
            minimum_bbox_supporting_line_distance=float(np.minimum(abs(uv),abs(uv-(args.n-1))).min()),
            white_gradient_norm=norm, probes=probes)

    rows=[]
    with threadpool_limits(limits=1):
        for repeat in range(args.warm_repeats+1):
            gc.collect()
            trial = root/("cold" if repeat==0 else f"warm_{repeat-1:02d}")
            trial.mkdir()
            timings={}
            print(f"[fit] {args.profile}/seed_{args.seed:02d}/{trial.name} Nd={projection.n_data} sigma={float(problem.noise_active[0])}", flush=True)
            begin=perf_counter()
            t=perf_counter()
            hs=source_metric(jnp.zeros(problem.n_source))
            cs=checked_factor(hs)
            source=jsl.cho_solve((cs,True),rhs)
            sync(source)
            equation_error=float(jnp.linalg.norm(jnp.asarray(hs)@source-rhs)/jnp.maximum(1.,jnp.linalg.norm(rhs)))
            if equation_error>1e-6:
                raise FloatingPointError(f"Source initialization normal equation residual {equation_error}")
            base=jnp.concatenate((source,jnp.zeros(problem.n_potential)))
            timings["source_initialization_seconds"]=perf_counter()-t
            del hs,cs
            t=perf_counter()
            h0=metric(base)
            timings["initial_metric_seconds"]=perf_counter()-t
            t=perf_counter()
            chol=checked_factor(h0)
            timings["initial_factor_seconds"]=perf_counter()-t
            t=perf_counter()
            f0,g0=gradient(base)
            initial_gamma=float(jnp.max(abs(g0)/jnp.sqrt(jnp.asarray(np.diag(h0)))))
            zero=jnp.zeros(problem.n_joint)
            sync(white_vg(zero,base,chol))
            timings["gradient_warmup_seconds"]=perf_counter()-t
            # Every fit constructs numerical factors again; no matrix cache.
            rng=np.random.default_rng(49)
            z=jnp.asarray(rng.normal(size=problem.n_joint))
            z=z/jnp.linalg.norm(z)
            delta=decode(z,base,chol)-base
            whitener_error=abs(float(delta@(jnp.asarray(h0)@delta))-1.)
            if whitener_error>1e-5:
                raise FloatingPointError(f"Whitening quadratic error {whitener_error}")
            del h0
            history=[]
            evaluations=0
            def fun(z):
                nonlocal evaluations
                value, grad=white_vg(jnp.asarray(z),base,chol)
                evaluations+=1
                return float(value),np.asarray(grad,dtype=np.float64)
            def callback(intermediate_result):
                history.append(dict(iteration=len(history)+1,objective=float(intermediate_result.fun)))
            t=perf_counter()
            result=minimize(fun,np.zeros(problem.n_joint),jac=True,method="L-BFGS-B",
                            options=OPTIONS.copy(),callback=callback)
            values=decode(jnp.asarray(result.x),base,chol)
            sync(values)
            timings["lbfgs_seconds"]=perf_counter()-t
            t=perf_counter()
            products=problem._result_products(values,ls,l4)
            sync(products)
            timings["products_seconds"]=perf_counter()-t
            timings["reconstruction_seconds"]=sum(timings.values())
            # Persist the raw endpoint even if a later qualification fails.
            np.savez_compressed(trial/"optimizer_state.npz",joint_values=np.asarray(values),
                initial_values=np.asarray(base),z=result.x)
            write(trial/"native_optimizer.json",dict(success=bool(result.success),status=int(result.status),
                message=str(result.message),iterations=int(result.nit),nfev=int(result.nfev),objective=float(result.fun)))
            # Research qualification is timed separately from reconstruction.
            t=perf_counter()
            f,g=gradient(values)
            he=metric(values)
            ce=checked_factor(he)
            proxy_g=jsl.solve_triangular(ce,g,lower=True)
            proxy=float(.5*jnp.vdot(proxy_g,proxy_g))
            gamma=float(jnp.max(abs(g)/jnp.sqrt(jnp.asarray(np.diag(he)))))
            analytic=problem._evaluate_kernel(values,ls,l4)
            relative_g_error=float(jnp.linalg.norm(g+analytic.negative_gradient)/jnp.maximum(1.,jnp.linalg.norm(g)))
            objective_error=abs(float(f-analytic.objective))
            timings["endpoint_qualification_seconds"]=perf_counter()-t
            del he,ce
            finite=bool(jnp.all(jnp.isfinite(values)) & jnp.isfinite(f))
            if relative_g_error>1e-7 or objective_error>1e-6:
                raise FloatingPointError("AD/analytical objective or gradient disagreement")
            core=np.asarray(grid.core_indices)
            kappa=np.asarray(products["convergence"])[core]
            peak_index=core[int(np.argmax(kappa))]
            residual=(problem.data_active-products["nonlinear_image_active"])/problem.noise_active
            threshold=cfg["iteration_options"]["gradient_atol"]+1e-8*initial_gamma
            row=dict(profile=args.profile,source_n=args.n,observation="binned",noise_seed=args.seed,
                target_sigma=protocol["target_sigma"],repeat=repeat,phase=trial.name,
                directory=str(trial.relative_to(args.output)),dimensions=problem._configuration(),
                native_success=bool(result.success),native_status=int(result.status),native_message=str(result.message),
                iterations=int(result.nit),nfev=int(result.nfev),objective_evaluations=evaluations,
                objective=float(f),initial_objective=float(f0),objective_decrease=float(f0-f),
                gn_stationarity_proxy=proxy,proxy_tolerance=.001,
                scaled_gradient=gamma,legacy_gradient_tolerance=threshold,legacy_gradient_pass=gamma<=threshold,
                accepted=bool(result.success and finite and proxy<=.001),
                source_initial_equation_error=equation_error,source_initial_hash=digest(base),
                whitening_quadratic_error=whitener_error,analytical_gradient_relative_error=relative_g_error,
                analytical_objective_absolute_error=objective_error,
                residual_rms_noise=float(jnp.sqrt(jnp.mean(residual**2))),
                kappa_peak=float(kappa.max()),kappa_peak_xy=[float(grid.x[peak_index]),float(grid.y[peak_index])],
                kappa_rms=float(np.sqrt(np.mean(kappa**2))),finite=finite,
                timings=timings,compute_seconds_excluding_io=perf_counter()-begin)
            write(trial/"optimizer.json",row)
            write(trial/"history.json",history)
            np.savez_compressed(trial/"fields.npz",joint_values=np.asarray(values),initial_values=np.asarray(base),
                source_values=np.asarray(products["source_values"]),source_x_axis=np.asarray(renderer.source_x_axis),
                source_y_axis=np.asarray(renderer.source_y_axis),convergence=np.asarray(products["convergence"]),
                model_image=np.asarray(products["nonlinear_model_image"]),fine_model_image=np.asarray(products["fine_model_image"]),
                observed_data=np.asarray(problem.data_active),observed_noise=np.asarray(problem.noise_active),
                normalized_residual=np.asarray(products["nonlinear_normalized_residual"]),
                observation_indices=np.asarray(projection.indices),observation_mask=projection.mask,
                potential_values=np.asarray(products["potential_values"]),
                potential_x=np.asarray(grid.x),potential_y=np.asarray(grid.y),potential_core_indices=core)
            if repeat==0:
                write(trial/"directional_profile.json",directional_profile(values,g,chol))
            rows.append(row)
            write(root/"summary.json",dict(rows=rows))
            print(f"[result] {trial.name} accepted={row['accepted']} nit={result.nit} "
                  f"proxy={proxy:.6g} inner={timings['lbfgs_seconds']:.3f}s "
                  f"reconstruction={timings['reconstruction_seconds']:.3f}s peak={row['kappa_peak']:.6g}",flush=True)
            del chol,source,base,values,products,g,g0,proxy_g,analytic,delta


def parent(args):
    import numpy as np
    from coarse_noise import add_matched_noise
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError("Use a fresh output directory")
    args.output.mkdir(parents=True,exist_ok=True)
    previous=json.loads((args.reference/"protocol.json").read_text())
    previous_summary=json.loads((args.reference/"summary.json").read_text())
    if not previous_summary["complete"]:
        raise ValueError("The previous coarse reference is incomplete")
    cfg=json.loads(CONFIG.read_text())
    if cfg!=previous["configuration"]:
        raise ValueError("Original Mock configuration changed")
    for name in ("frozen_inputs.npz","frozen_inputs.json","postfit_truth.npz"):
        shutil.copy2(args.reference/name,args.output/name)
    with np.load(args.reference/"baseline/binned/cold/fields.npz") as a:
        base_data=a["observed_data"].copy();base_sigma=a["observed_noise"].copy()
        indices=a["observation_indices"].copy();mask=a["observation_mask"].copy()
    with np.load(args.output/"frozen_inputs.npz") as a:
        active=~a["mask"]
        expected=a["image_data"][active][indices].mean(axis=1)
        expected_sigma=np.sqrt((a["noise_map"][active][indices]**2).sum(axis=1))/indices.shape[1]
        clean=a["model_image"][active][indices].mean(axis=1)
    np.testing.assert_allclose(base_data,expected,rtol=1e-14,atol=1e-14)
    np.testing.assert_allclose(base_sigma,expected_sigma,rtol=1e-14,atol=1e-14)
    if len(base_data)!=2312 or indices.shape!=(2312,9) or previous["factor"]!=3:
        raise ValueError("This stress test requires the fixed 2312-point 3x3 reference")
    np.savez_compressed(args.output/"coarse_base.npz",data=base_data,sigma=base_sigma,
        indices=indices,mask=mask,clean_signal=clean,original_noise=base_data-clean)
    inputs={}
    for seed in args.seeds:
        noisy=add_matched_noise(base_data,base_sigma,args.target_sigma,seed)
        path=args.output/f"noise_seed_{seed:02d}.npz"
        np.savez_compressed(path,**noisy)
        inputs[str(seed)]=dict(sha256=sha(path),seed=seed,points=len(base_data),
            added_sigma_min=float(noisy["extra_sigma"].min()),added_sigma_max=float(noisy["extra_sigma"].max()),
            total_noise_realized_rms=float(np.sqrt(np.mean((noisy["data"]-clean)**2))),
            total_noise_realized_mean=float(np.mean(noisy["data"]-clean)),
            standard_normal_realized_rms=float(np.sqrt(np.mean(noisy["extra_standard_normal"]**2))))
    sources=[Path(__file__),HERE/"coarse_noise.py",HERE/"data_resolution_observation.py",
        HERE/"source_resolution_prior.py",HERE/"run_source_resolution.py",HERE/"run_experiment.py",
        HERE/"run_scientific.py",CONFIG,ROOT/"examples/potential_correction/mock.py",
        ROOT/"TinyLensGpu/ForwardSimulation/LensImage/potential_correction.py",
        ROOT/"TinyLensGpu/ObservationModel/LensImage/_potential_correction_common.py",
        ROOT/"TinyLensGpu/ObservationModel/LensImage/potential_correction_iterative.py",
        ROOT/"TinyLensGpu/utils/lensing/mapping.py",ROOT/"TinyLensGpu/utils/lensing/potential.py",
        ROOT/"TinyLensGpu/utils/inversion/regularization.py",
        ROOT/"TinyLensGpu/utils/inversion/potential_regularization.py"]
    for path in sources:
        target=args.output/"source_snapshot"/path.relative_to(ROOT)
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)
    write(args.output/"protocol.json",dict(reference=str(args.reference.resolve()),
        configuration=cfg,profiles=previous["profiles"],source_grid=100,source_bbox=cfg["source_bbox"],
        source_prior=previous["source_prior"],factor=3,data_points=2312,potential_parameters=956,
        source_parameters=10000,total_parameters=10956,target_sigma=args.target_sigma,
        old_sigma_min=float(base_sigma.min()),old_sigma_max=float(base_sigma.max()),
        noise_inputs=inputs,extra_noise_seeds=args.seeds,
        noise_construction="old coarse data + sqrt(target_sigma^2 - old_sigma^2) * independent N(0,1)",
        noise_seed_interpretation="Independent added components; final data share the original seed17 coarse noise",
        cross_seed_shared_variance_fraction=float(np.mean(base_sigma**2)/args.target_sigma**2),
        fixed_state_data_Fisher_factor=float(np.mean(base_sigma**2)/args.target_sigma**2),
        physical_operator="Same 3x3 block mean after original fine PSF; every fine ray and potential node unchanged",
        initialization=previous["initialization"],preconditioning=previous["preconditioning"],
        options=OPTIONS,qualification=previous["qualification"],warm_repeats=args.warm_repeats,
        evidence_optimization=False,lambda_retuning=False,dtype="float64",
        hypotheses=[
            "Reducing SNR produces J0946-like failure of the same GN stationarity criterion",
            "The fixed prior suppresses the correction more strongly while the numerical solver still passes",
            "Different added-noise seeds expose failure or physical recovery variation beyond technical repeats"],
        reconstruction_timing=previous["reconstruction_timing"],source_truth_used_for_initialization=False,
        truth_only_for_postfit_diagnostics=True,
        reference_files_sha256={name:sha(args.reference/name) for name in [
            "protocol.json","analysis.json","summary.json","postfit_truth.npz",
            "baseline/binned/summary.json","evidence_selected/binned/summary.json"]},
        frozen_input_sha256=sha(args.output/"frozen_inputs.npz"),
        coarse_base_sha256=sha(args.output/"coarse_base.npz"),
        truth_sha256=sha(args.output/"postfit_truth.npz"),
        sources_sha256={str(p.relative_to(ROOT)):sha(p) for p in sources}))
    env=os.environ.copy();env.update(OPENBLAS_NUM_THREADS="1",OMP_NUM_THREADS="1",MKL_NUM_THREADS="1")
    statuses=[];started=perf_counter()
    cases=[(seed,p) for i,seed in enumerate(args.seeds)
           for p in (list(previous["profiles"]) if i%2==0 else list(reversed(previous["profiles"])))]
    for seed,profile in cases:
        name=f"{profile}_seed_{seed:02d}";print(f"[process] {name}",flush=True)
        command=[sys.executable,str(Path(__file__).resolve()),"--worker",
            "--output",str(args.output.resolve()),"--reference",str(args.reference.resolve()),
            "--profile",profile,"--seed",str(seed),"--target-sigma",str(args.target_sigma),
            "--warm-repeats",str(args.warm_repeats),"--panel",str(args.panel)]
        with (args.output/f"{name}.log").open("w") as handle:
            proc=subprocess.run(command,stdout=handle,stderr=subprocess.STDOUT,env=env,cwd=ROOT)
        statuses.append(dict(condition=name,returncode=proc.returncode))
        write(args.output/"process_status.json",statuses)
        print(f"[process_done] {name} returncode={proc.returncode}",flush=True)
    groups={}
    for seed,profile in cases:
        path=args.output/profile/f"seed_{seed:02d}"/"summary.json"
        if path.exists():groups[f"{profile}/seed_{seed:02d}"]=json.loads(path.read_text())
    write(args.output/"summary.json",dict(groups=groups,process_status=statuses,
        elapsed_seconds=perf_counter()-started,complete=all(s["returncode"]==0 for s in statuses),
        frozen_input_sha256=sha(args.output/"frozen_inputs.npz")))
    if any(s["returncode"]!=0 for s in statuses):
        raise RuntimeError("A condition failed; raw logs, endpoints and partial summaries are preserved")
    print(f"[done] {args.output}",flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=HERE/"coarse_noise_v1")
    parser.add_argument("--reference",type=Path,default=HERE/"data_resolution_v2")
    parser.add_argument("--seeds",nargs="+",type=int,default=[0,1,2])
    parser.add_argument("--seed",type=int,default=0)
    parser.add_argument("--target-sigma",type=float,default=.01)
    parser.add_argument("--warm-repeats",type=int,default=3)
    parser.add_argument("--panel",type=int,default=16)
    parser.add_argument("--profile",choices=("baseline","evidence_selected"),default="baseline")
    parser.add_argument("--worker",action="store_true")
    args=parser.parse_args()
    args.n=100;args.factor=3
    if args.warm_repeats<1 or args.panel<1 or len(set(args.seeds))!=len(args.seeds):
        raise ValueError("Use positive repeat/panel counts and distinct noise seeds")
    if args.worker:worker(args)
    else:parent(args)


if __name__=="__main__":
    main()

