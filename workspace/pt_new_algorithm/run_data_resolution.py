"""Paired image-binning experiment at fixed 100x100 source and potential grid.

The actual PSF, pixel integration and potential geometry stay on the fine
grid. A disjoint block mean is applied to both noisy data and predictions.
Full, footprint-matched fine, and coarse views isolate boundary selection
from spatial resolution; all numerical matrices are rebuilt for each fit.
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

from run_source_resolution import ROOT, HERE, CONFIG, OPTIONS, write, sha, runtime


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
    coarse=ObservationProjection(saved["mask"],args.factor)
    projection=ObservationProjection(saved["mask"]) if args.observation=="full" else (
        ObservationProjection(saved["mask"],matched_to=coarse) if args.observation=="matched_fine" else coarse)
    problem = ProjectedPotentialCorrection(renderer, obs.image_data, obs.noise_map,
        jnp.zeros(renderer.n_source), lambda_amplitude=l0, source_regularization="second-order",
        solve_rtol=cfg["solve_rtol"], solve_atol=cfg["solve_atol"], projection=projection)
    # Replace before tracing any kernels. No core-library defaults are changed.
    problem.source_regularizer = PhysicalCurvaturePrior(args.n, reference_n=49)
    problem.initialization_provenance.update(source_truth_used=False,
        source_initialization="data_only_source_MAP_at_zero_potential",
        macro_fixed_to_truth=True, source_prior="physical_integrals_matched_to_n49")
    root = args.output/args.profile/args.observation
    root.mkdir(parents=True)
    write(root/"environment.json", dict(python=platform.python_version(), jax=jax.__version__,
        numpy=np.__version__, scipy=scipy.__version__, devices=[d.device_kind for d in jax.devices()],
        dimensions=problem._configuration(), source_spacing_ratio=problem.source_regularizer.spacing_ratio,
        observation=args.observation,projection_shape=projection.shape,n_observations=projection.n_data,
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
            print(f"[fit] {args.profile}/{args.observation}/{trial.name} Nd={projection.n_data}", flush=True)
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
            row=dict(profile=args.profile,source_n=args.n,observation=args.observation,repeat=repeat,phase=trial.name,
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
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError("Use a fresh output directory")
    args.output.mkdir(parents=True,exist_ok=True)
    reference=json.loads((args.reference/"protocol.json").read_text())
    reference_summary=json.loads((args.reference/"summary.json").read_text())
    if not reference_summary["complete"]:
        raise ValueError("The frozen reference experiment must be complete")
    for name in ("frozen_inputs.npz","frozen_inputs.json"):
        shutil.copy2(args.reference/name,args.output/name)
    cfg=json.loads(CONFIG.read_text())
    if cfg!=reference["configuration"]:
        raise ValueError("Reference Mock configuration changed")
    sources=[Path(__file__),HERE/"data_resolution_observation.py",HERE/"source_resolution_prior.py",
        HERE/"run_source_resolution.py",HERE/"run_experiment.py",HERE/"run_scientific.py",CONFIG,
        ROOT/"TinyLensGpu/ForwardSimulation/LensImage/potential_correction.py",
        ROOT/"TinyLensGpu/ObservationModel/LensImage/_potential_correction_common.py",
        ROOT/"TinyLensGpu/ObservationModel/LensImage/potential_correction_iterative.py",
        ROOT/"TinyLensGpu/utils/lensing/mapping.py",ROOT/"TinyLensGpu/utils/lensing/potential.py",
        ROOT/"TinyLensGpu/utils/inversion/regularization.py",
        ROOT/"TinyLensGpu/utils/inversion/potential_regularization.py"]
    for path in sources:
        target=args.output/"source_snapshot"/path.relative_to(ROOT)
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)
    write(args.output/"protocol.json",dict(
        reference=str(args.reference.resolve()),configuration=cfg,profiles=reference["profiles"],
        source_grid=100,source_bbox=cfg["source_bbox"],source_prior=reference["source_prior"],
        factor=args.factor,observations=["full","matched_fine","binned"],
        data="Same frozen noisy realization; means of complete disjoint blocks only",
        noise="sqrt(sum(sigma_i^2))/K; no overlapping blocks, so covariance remains diagonal",
        PSF="Exact original fine-grid convolution followed by the same projection as the data",
        potential_grid="All original nodes, spacing, derivative stencils and fine queries unchanged",
        initialization=reference["initialization"],preconditioning=reference["preconditioning"],
        options=OPTIONS,qualification=reference["qualification"],warm_repeats=args.warm_repeats,
        evidence_optimization=False,dtype="float64",
        hypotheses=[
            "If data-to-parameter count alone causes failure, coarse fits will miss the same .001 GN gate",
            "If information loss mainly changes cost/uncertainty, both fine and binned fits can still pass",
            "If the mask crop matters, matched_fine already changes convergence before binning"],
        reconstruction_timing=reference["reconstruction_timing"],
        source_truth_used_for_initialization=False,
        reference_protocol_sha256=sha(args.reference/"protocol.json"),
        frozen_input_sha256=sha(args.output/"frozen_inputs.npz"),
        sources_sha256={str(p.relative_to(ROOT)):sha(p) for p in sources}))
    env=os.environ.copy();env.update(OPENBLAS_NUM_THREADS="1",OMP_NUM_THREADS="1",MKL_NUM_THREADS="1")
    statuses=[];started=perf_counter()
    cases=[(profile,obs) for profile in reference["profiles"]
           for obs in (["full","matched_fine","binned"] if profile=="baseline" else ["binned","matched_fine","full"])]
    for profile,obs in cases:
        name=f"{profile}_{obs}";print(f"[process] {name}",flush=True)
        command=[sys.executable,str(Path(__file__).resolve()),"--worker",
            "--output",str(args.output.resolve()),"--reference",str(args.reference.resolve()),
            "--profile",profile,"--observation",obs,"--n","100","--factor",str(args.factor),
            "--panel",str(args.panel),"--warm-repeats",str(args.warm_repeats)]
        with (args.output/f"{name}.log").open("w") as handle:
            proc=subprocess.run(command,stdout=handle,stderr=subprocess.STDOUT,env=env,cwd=ROOT)
        statuses.append(dict(condition=name,returncode=proc.returncode))
        write(args.output/"process_status.json",statuses)
        print(f"[process_done] {name} returncode={proc.returncode}",flush=True)
    groups={}
    for profile,obs in cases:
        path=args.output/profile/obs/"summary.json"
        if path.exists():groups[f"{profile}/{obs}"]=json.loads(path.read_text())
    write(args.output/"summary.json",dict(groups=groups,process_status=statuses,
        elapsed_seconds=perf_counter()-started,complete=all(s["returncode"]==0 for s in statuses),
        frozen_input_sha256=sha(args.output/"frozen_inputs.npz")))
    if any(s["returncode"]!=0 for s in statuses):
        raise RuntimeError("At least one condition failed; raw logs and partial summaries are preserved")
    print(f"[done] {args.output}",flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=HERE/"data_resolution_v1")
    parser.add_argument("--reference",type=Path,default=HERE/"source_resolution_v2")
    parser.add_argument("--warm-repeats",type=int,default=3)
    parser.add_argument("--panel",type=int,default=16)
    parser.add_argument("--n",type=int,choices=(100,),default=100)
    parser.add_argument("--factor",type=int,default=3)
    parser.add_argument("--profile",choices=("baseline","evidence_selected"),default="baseline")
    parser.add_argument("--observation",choices=("full","matched_fine","binned"),default="full")
    parser.add_argument("--worker",action="store_true")
    args=parser.parse_args()
    if args.warm_repeats<1 or args.panel<1 or args.factor<2:
        raise ValueError("Use positive repeat/panel counts and factor >=2")
    if args.worker:worker(args)
    else:parent(args)


if __name__=="__main__":
    main()

