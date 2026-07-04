"""
Adaptive-regularization inference pipeline for the pix_src_pipe demo (Bspline light models).

Stage a  : SIE + shear + Bspline lens light + Bspline source light (uniform priors)
Stage b  : build an arc feature mask from stage-A residuals
Stage l  : arc-masked Bspline lens light refinement (Gaussian priors from stage a)
Stage m0 : SIE + shear + uniform pixelized source — GPU grid search for
            evidence-best lambda_reg on lens-subtracted image, builds the fixed S0 source template
Stage m1 : EPL + shear + non-adaptive pixelized source — Nautilus sampling
            of mass parameters on lens-subtracted image with lambda_reg constrained around stage-M0,
            then builds the fixed S1 source template
Stage m2 : fixed EPL + shear + adaptive pixelized source — Nautilus sampling
            of log_lambda_reg and adaptive_reg_rho only, on lens-subtracted image
Stage m3 : EPL + shear + adaptive pixelized source — Nautilus nested sampling
            with source-reg hyperparameters fixed from stage-M2 medians

Each stage pickles its posterior samples/weights to
``output/stage_{a,l,m0,m1,m2,m3}.pkl`` and is re-runnable via ``--skip-done``.
"""

from __future__ import annotations

import argparse
import os
import pickle
import time
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("NVIDIA_TF32_OVERRIDE", "0")

os.chdir(Path(__file__).parent)

import jax
jax.config.update("jax_default_matmul_precision", "float32")

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from nautilus import Sampler

from TinyLensGpu.Inference import ParamU
from TinyLensGpu.Inference.build_likelihood import make_likelihood
from TinyLensGpu.Inference.build_prior import make_prior_transformation
from TinyLensGpu.ObservationModel import PointSourceProbModel
from TinyLensGpu.ObservationModel.LensImage.parametric_image_model import ImageProbModel
from TinyLensGpu.ObservationModel.LensImage.pixelized_image_model_operator import PixelizedImageProbModelOperator
from TinyLensGpu.PhysicalModel import (
    PhysicalModel,
    PixelizedSourceModel,
    Shear,
)
from TinyLensGpu.PhysicalModel.LensImage.Parametric.Light import build_bspline_multipole_set
from TinyLensGpu.PhysicalModel.LensImage.Parametric.Mass import SIE, EPL
from TinyLensGpu.utils import load_lens_data
from TinyLensGpu.utils.inversion.regularization import source_template_scale_map
from TinyLensGpu.utils.misc import arc_mask_from, weighted_quantile
from TinyLensGpu.visualizer import plot_model_results, overlay_critical_and_caustics

from TinyLensGpu.Inference import StagePosterior

import caskade as ck
import jax.scipy.linalg as jsl

# ------------------------------------------------------------------ #
NSRC = 30
DPIX = 0.05
NSUB = 4
NSUB_PIX = 4
MASK_RADIUS = 2.5
ADAPTIVE_REG_RHO = 2.0
ADAPTIVE_REG_RHO_PRIOR_MAX = 8.0
PIXEL_REGULARIZATION_TYPE = "first-order"
SOURCE_BBOX_PADDING = 0.30
FISTA_MAX_ITER = 1000
FISTA_RTOL = 1.0e-5
FISTA_POWER_ITER = 10
FISTA_STEP_SAFETY = 1.2
SOLVER_TYPE = "pcg"
OUT_DIR = Path("output")
DATA_DIR = Path("data")


# ------------------------------------------------------------------ #
def _run_sampler(likelihood, n_live: int, n_eff: int, tag: str, vectorized: bool = True):
    prior, prior_specs = make_prior_transformation(likelihood)
    param_names = [spec.name for spec in prior_specs]
    print(f"\n[{tag}] {len(param_names)} dynamic params:")
    for spec in prior_specs:
        print(f"    {spec.name:25s} {spec.describe()}")
    loglike = make_likelihood(likelihood, vectorized=vectorized)
    sampler_kwargs = dict(n_live=n_live, vectorized=vectorized)
    if vectorized: sampler_kwargs["n_batch"] = 100
    sampler = Sampler(prior, loglike, n_dim=len(param_names), **sampler_kwargs)
    sampler.run(verbose=True, n_eff=n_eff)
    samples, log_w, _ = sampler.posterior()
    samples = np.asarray(samples, dtype=np.float64)
    weights = np.exp(np.asarray(log_w, dtype=np.float64)); weights /= weights.sum()
    log_z = float(sampler.log_z)
    print(f"[{tag}] log_z = {log_z:.3f}")
    return StagePosterior.from_likelihood(likelihood, samples, weights, log_z=log_z)


def _dump_stage(tag: str, samples, weights, param_names, log_z, extra=None, stage=None):
    OUT_DIR.mkdir(exist_ok=True)
    if stage is not None:
        sp = stage.cache_payload()
        payload = dict(samples=sp["samples"], weights=sp["weights"], param_names=sp["param_names"], log_z=sp["log_z"],
                       stage_schema={k:sp[k] for k in ("param_names","prior_specs") if k in sp}, extra=extra or {})
    else:
        payload = dict(samples=samples, weights=weights, param_names=param_names, log_z=log_z,
                       stage_schema={"param_names": list(param_names)}, extra=extra or {})
    with open(OUT_DIR / f"stage_{tag}.pkl", "wb") as f: pickle.dump(payload, f)
    print(f"[{tag}] posterior saved to {OUT_DIR}/stage_{tag}.pkl")


def _load_stage(tag: str):
    with open(OUT_DIR / f"stage_{tag}.pkl", "rb") as f: return pickle.load(f)


def _stage_from_payload(payload):
    s = payload.get("stage_schema", {})
    return StagePosterior.from_schema(payload["samples"], payload["weights"],
                                       prior_specs=payload.get("prior_specs") or s.get("prior_specs"),
                                       param_names=None if (payload.get("prior_specs") or s.get("prior_specs")) else (payload.get("param_names") or s.get("param_names")),
                                       log_z=payload.get("log_z"))


def _print_summary(tag, samples, weights, param_names):
    print(f"\n[{tag}] Posterior summary:")
    q = np.array([0.16, 0.5, 0.84])
    for i, n in enumerate(param_names):
        qs = weighted_quantile(np.asarray(samples[:, i]), weights, q)
        print(f"    {n:25s} = {float(qs[1]):+.4f} ({float(qs[0])-float(qs[1]):+.4f}, {float(qs[2])-float(qs[1]):+.4f})")


# ------------------------------------------------------------------ #
def _make_circular_mask(image_shape, dpix, radius_arcsec=3.5):
    ny, nx = image_shape; y = (np.arange(ny)-(ny-1)/2)*dpix; x = (np.arange(nx)-(nx-1)/2)*dpix
    yy, xx = np.meshgrid(y, x, indexing="ij"); return (xx**2+yy**2) > radius_arcsec**2


def _source_axes_from_bbox(source_bbox, n=NSRC):
    xmin,xmax,ymin,ymax = [float(v) for v in source_bbox]
    return (np.linspace(xmin,xmax,int(n),dtype=np.float64), np.linspace(ymin,ymax,int(n),dtype=np.float64))


def _is_square_bbox(source_bbox, *, rtol=1.0e-6, atol=1.0e-7):
    xmin,xmax,ymin,ymax = [float(v) for v in source_bbox]; return np.isclose(xmax-xmin,ymax-ymin,rtol=rtol,atol=atol)


def _make_s0_scale(s0_package, rho=ADAPTIVE_REG_RHO):
    return source_template_scale_map(s0_package["source_pixels"], int(s0_package["n"]), rho=rho)


def _validate_s0_package(s0_package):
    if [k for k in ("nx","ny") if k in s0_package]:
        raise KeyError("S0 package uses legacy source-grid keys")
    required = ("source_pixels","source_bbox","source_x_axis","source_y_axis","n","lambda_best","log_lambda_best")
    missing = [k for k in required if k not in s0_package]
    if missing: raise KeyError("S0 package missing: "+", ".join(missing))
    n=int(s0_package["n"])
    if n!=NSRC: raise ValueError(f"S0 n={n} != configured {NSRC}")
    sp=np.asarray(s0_package["source_pixels"])
    if sp.shape!=(n*n,): raise ValueError(f"S0 source_pixels shape {sp.shape} != ({n*n},)")
    for an in ("source_x_axis","source_y_axis"):
        a=np.asarray(s0_package[an])
        if a.shape!=(n,): raise ValueError(f"S0 {an} shape {a.shape} != ({n},)")
    bbox=tuple(float(v) for v in s0_package["source_bbox"])
    if len(bbox)!=4 or not np.all(np.isfinite(bbox)): raise ValueError("S0 source_bbox invalid")
    if not (bbox[0]<bbox[1] and bbox[2]<bbox[3]): raise ValueError("S0 source_bbox not sorted")
    if not _is_square_bbox(bbox): raise ValueError("S0 source_bbox not square")
    sm=s0_package.get("scale_map")
    if sm is None:
        sm=np.asarray(_make_s0_scale(s0_package),dtype=np.float32); s0_package["scale_map"]=sm
    else:
        sm=np.asarray(sm,dtype=np.float32)
        if sm.shape!=(n*n,): raise ValueError(f"S0 scale_map shape {sm.shape} != ({n*n},)")
        if not np.all(np.isfinite(sm)&(sm>0.0)): raise ValueError("S0 scale_map invalid")
        s0_package["scale_map"]=sm
    return s0_package


def _fista_kwargs():
    return dict(fista_max_iter=FISTA_MAX_ITER, fista_rtol=FISTA_RTOL, fista_power_iter=FISTA_POWER_ITER, fista_step_safety=FISTA_STEP_SAFETY)


def _s0_fixed_kwargs(s0_package):
    s0_package=_validate_s0_package(s0_package)
    return dict(fixed_source_bbox=tuple(float(v) for v in s0_package["source_bbox"]),
                fixed_reg_template=jnp.asarray(s0_package["source_pixels"],dtype=jnp.float32))


def _source_param_value(value): return value.value if hasattr(value,"value") else value


def _reg_hyperparams_from_payload(sp, tag):
    medians=sp.get("extra",{}).get("medians")
    if not medians: raise KeyError(f"stage-{tag.upper()} payload missing medians")
    required=("log_lambda_reg","adaptive_reg_rho")
    missing=[n for n in required if n not in medians]
    if missing: raise KeyError(f"stage-{tag.upper()} medians missing: "+", ".join(missing))
    return {n:float(medians[n]) for n in required}


def _reg_hyperparams_from_m2_payload(sp): return _reg_hyperparams_from_payload(sp,"m2")


def _format_reg_hyperparams(rh):
    return f"lambda={float(jnp.exp(rh['log_lambda_reg'])):.4e}, rho={rh['adaptive_reg_rho']:.4f}"


def _valid_log_evidence(values) -> np.ndarray:
    v=np.asarray(values,dtype=np.float64); return np.isfinite(v)&(v>-1.0e9)


def _solve_pixel_source_for_package(likelihood, medians, param_names):
    q50=[medians[n] for n in param_names]
    with ck.ActiveContext(likelihood):
        likelihood.fill_params(jnp.array(q50))
        lj=jnp.exp(likelihood.phys_model.source_light[0].log_lambda_reg.value)
        (xmin,xmax,ymin,ymax,bx_sub,by_sub,bx_seed,by_seed)=likelihood._get_bbox()
        scale=likelihood._get_reg_scale()
        rd=likelihood._regularization_data(xmin,xmax,ymin,ymax,scale=scale)
        od=likelihood.sim_obj.precompute_operator_data(xmin,xmax,ymin,ymax,_betas_sub=(bx_sub,by_sub))
        bc,bm=likelihood.sim_obj.build_block_diag_preconditioner(
            likelihood.noise_1d,xmin,xmax,ymin,ymax,lj,likelihood.reg_builder,block_size=likelihood.block_size,scale=scale)
        sp,si=likelihood._solve_source(xmin,xmax,ymin,ymax,lj,rd,(bc,bm),op_data=od)
        if not bool(np.asarray(si.converged)):
            ml="residual" if hasattr(si,"residual_norm") else "convergence_metric"
            mv=float(si.residual_norm) if hasattr(si,"residual_norm") else float(si.convergence_metric)
            raise RuntimeError(f"{likelihood.solver_type.upper()} failed ({ml}={mv:.4e}, n_iter={int(si.n_iter)})")
    sbbox=(float(xmin),float(xmax),float(ymin),float(ymax))
    xa,ya=_source_axes_from_bbox(sbbox,NSRC)
    return dict(source_pixels=np.asarray(sp,dtype=np.float64), source_image=np.asarray(sp,dtype=np.float64).reshape(NSRC,NSRC),
                source_bbox=sbbox, source_x_axis=xa, source_y_axis=ya, n=NSRC)


# ------------------------------------------------------------------ #
# Stage A — SIE + shear + Bspline lens light + Bspline source light
# ------------------------------------------------------------------ #
def build_stage_a_likelihood(image_data, noise_map, psf_kernel, mask=None):
    sie = SIE(
        theta_E=ParamU("theta_E",1.5,prior_type="uniform",prior_settings=[0.5,2.5],limits=[0.0,5.0]),
        e1=ParamU("e1_mass",0.0,prior_type="gaussian",prior_settings=[0.0,0.3],limits=[-1.0,1.0]),
        e2=ParamU("e2_mass",0.0,prior_type="gaussian",prior_settings=[0.0,0.3],limits=[-1.0,1.0]),
        center_x=ParamU("center_x_mass",0.0,prior_type="gaussian",prior_settings=[0.0,0.1],limits=[-1.0,1.0]),
        center_y=ParamU("center_y_mass",0.0,prior_type="gaussian",prior_settings=[0.0,0.1],limits=[-1.0,1.0]),
    )
    shear = Shear(
        gamma1=ParamU("gamma1",0.0,prior_type="uniform",prior_settings=[-0.2,0.2],limits=[-0.5,0.5]),
        gamma2=ParamU("gamma2",0.0,prior_type="uniform",prior_settings=[-0.2,0.2],limits=[-0.5,0.5]),
    )
    for p in (sie.theta_E,sie.e1,sie.e2,sie.center_x,sie.center_y,shear.gamma1,shear.gamma2): p.to_dynamic()

    # Bspline source
    cx_s=ParamU("center_x_src",0.0,prior_type="gaussian",prior_settings=[0.0,0.5],limits=[-3.0,3.0])
    cy_s=ParamU("center_y_src",0.0,prior_type="gaussian",prior_settings=[0.0,0.5],limits=[-3.0,3.0])
    e1_s=ParamU("e1_src",0.0,prior_type="gaussian",prior_settings=[0.0,0.3],limits=[-1.0,1.0])
    e2_s=ParamU("e2_src",0.0,prior_type="gaussian",prior_settings=[0.0,0.3],limits=[-1.0,1.0])
    source_bspline=build_bspline_multipole_set(dpix=DPIX,r_min=0.001,r_max=1.0,n_radial=20,ntheta=[0],degree=3,
                                                center_x=cx_s,center_y=cy_s,e1=e1_s,e2=e2_s,mask=None)
    cx_s.to_dynamic(); cy_s.to_dynamic(); e1_s.to_dynamic(); e2_s.to_dynamic()

    # Bspline lens light
    cx_l=ParamU("center_x_lens",0.0,prior_type="gaussian",prior_settings=[0.0,0.1],limits=[-1.0,1.0])
    cy_l=ParamU("center_y_lens",0.0,prior_type="gaussian",prior_settings=[0.0,0.1],limits=[-1.0,1.0])
    e1_l=ParamU("e1_lens",0.0,prior_type="gaussian",prior_settings=[0.0,0.3],limits=[-1.0,1.0])
    e2_l=ParamU("e2_lens",0.0,prior_type="gaussian",prior_settings=[0.0,0.3],limits=[-1.0,1.0])
    lens_bspline=build_bspline_multipole_set(dpix=DPIX,r_min=0.001,r_max=2.5,n_radial=20,ntheta=[0],degree=3,
                                              center_x=cx_l,center_y=cy_l,e1=e1_l,e2=e2_l,mask=None)
    cx_l.to_dynamic(); cy_l.to_dynamic(); e1_l.to_dynamic(); e2_l.to_dynamic()

    phys=PhysicalModel(lens_mass=[sie,shear],source_light=source_bspline,lens_light=lens_bspline)
    return ImageProbModel(image_data=image_data,noise_map=noise_map,psf_kernel=psf_kernel,
                           dpix=DPIX,nsub=NSUB,phys_model=phys,use_linear=True,solver_type="normal",mask=mask)


def run_stage_a(image_data, noise_map, psf_kernel, circular_mask=None):
    OUT_DIR.mkdir(exist_ok=True)
    print("\n"+"="*60); print(" Stage A : SIE + shear + Bspline lens light + Bspline source light"); print("="*60)
    t0=time.time()
    likelihood=build_stage_a_likelihood(image_data,noise_map,psf_kernel,mask=circular_mask)
    stage=_run_sampler(likelihood,n_live=200,n_eff=2000,tag="stage-A",vectorized=True)
    t1=time.time()
    samples,weights,names,logz=stage.samples,stage.weights,stage.param_names,stage.log_z
    _print_summary("stage-A",samples,weights,names); print(f"[stage-A] time taken: {t1-t0:.2f} seconds")
    medians=stage.medians(); q50=[medians[n] for n in names]
    likelihood.set_values(q50)
    fwd=likelihood.forward_model(use_linear=True,return_intensity=True,ret_each_plane=True,
                                  image_map=likelihood.image_data,noise_map=likelihood.noise_map)
    lens_image_model=np.asarray(fwd[1])
    _dump_stage("a",samples,weights,names,logz,
                extra=dict(medians=medians,lens_light_model=lens_image_model,time_taken=t1-t0),stage=stage)
    try:
        plot_model_results(likelihood,jnp.asarray(q50),save_path=str(OUT_DIR/"stage_a_model.png"),
                          title="Stage A : Bspline lens+source",show_critical_lines=True,show_caustics=True)
    except Exception as err: print(f"[stage-A] plotting failed (non-fatal): {err}")
    return stage,medians,lens_image_model


# ------------------------------------------------------------------ #
# Stage B
# ------------------------------------------------------------------ #
def run_stage_b(image_data, noise_map, lens_light_model, circular_mask=None):
    print("\n"+"="*60); print(" Stage B : build arc feature mask from stage-A residuals"); print("="*60)
    residual=image_data-lens_light_model; snr_map=residual/noise_map
    arc_mask=arc_mask_from(snr_map,threshold=3.0,ignor_size=25,ext_size=5,close_size=3)
    feature_mask=arc_mask
    if circular_mask is not None: feature_mask=feature_mask|circular_mask
    print(f"[stage-B] arc pixels kept = {int((~feature_mask).sum())} / {feature_mask.size}")
    DATA_DIR.mkdir(exist_ok=True); fits.writeto(DATA_DIR/"feature_mask.fits",feature_mask.astype(np.uint8),overwrite=True)
    ny,nx=image_data.shape; ext=[-nx*DPIX/2,nx*DPIX/2,-ny*DPIX/2,ny*DPIX/2]
    fig,axes=plt.subplots(1,2,figsize=(9,4))
    axes[0].imshow(residual,origin="lower",extent=ext,cmap="viridis"); axes[0].set_title("residual = image - stage-A lens light")
    axes[1].imshow(snr_map,origin="lower",extent=ext,cmap="viridis",vmin=-3,vmax=np.nanpercentile(snr_map,99.5))
    axes[1].set_title("S/N map + arc mask boundary")
    axes[1].contour(~feature_mask,levels=[0.5],origin="lower",extent=ext,colors="red",linewidths=1.5)
    plt.tight_layout(); OUT_DIR.mkdir(exist_ok=True)
    plt.savefig(OUT_DIR/"stage_b_mask.png",dpi=120,bbox_inches="tight"); plt.close(fig)
    return feature_mask


# ------------------------------------------------------------------ #
# Stage L — arc-masked Bspline lens light refinement
# ------------------------------------------------------------------ #
def build_stage_l_likelihood(image_data, noise_map, psf_kernel, feature_mask,
                               stage_a: StagePosterior, circular_mask=None):
    cx_l=stage_a.gaussian("center_x_lens",model="Gaussian",attr="center_x",limits=[-1.0,1.0])
    cy_l=stage_a.gaussian("center_y_lens",model="Gaussian",attr="center_y",limits=[-1.0,1.0])
    e1_l=stage_a.gaussian("e1_lens",model="Gaussian",attr="e1",limits=[-1.0,1.0])
    e2_l=stage_a.gaussian("e2_lens",model="Gaussian",attr="e2",limits=[-1.0,1.0])
    lens_bspline=build_bspline_multipole_set(dpix=DPIX,r_min=0.01,r_max=2.5,n_radial=15,ntheta=[0],degree=3,
                                              center_x=cx_l,center_y=cy_l,e1=e1_l,e2=e2_l,mask=None)
    cx_l.to_dynamic(); cy_l.to_dynamic(); e1_l.to_dynamic(); e2_l.to_dynamic()
    noise_map_soft=np.array(noise_map,copy=True); noise_map_soft[~feature_mask]*=1000.0
    phys=PhysicalModel(lens_mass=[],source_light=[],lens_light=lens_bspline)
    return ImageProbModel(image_data=image_data,noise_map=noise_map_soft,psf_kernel=psf_kernel,
                           dpix=DPIX,nsub=NSUB,phys_model=phys,use_linear=True,solver_type="normal",mask=circular_mask)


def run_stage_l(image_data, noise_map, psf_kernel, feature_mask,
                stage_a: StagePosterior, circular_mask=None):
    print("\n"+"="*60); print(" Stage L : arc-masked Bspline lens light refinement"); print("="*60)
    t0=time.time()
    na=int((~feature_mask).sum()); nn=feature_mask.size-na
    print(f"[stage-L] arc pixels excluded     = {na}    / {feature_mask.size}")
    print(f"[stage-L] lens-light pixels kept  = {nn} / {feature_mask.size}")
    likelihood=build_stage_l_likelihood(image_data,noise_map,psf_kernel,feature_mask,stage_a,circular_mask=circular_mask)
    stage=_run_sampler(likelihood,n_live=200,n_eff=800,tag="stage-L",vectorized=True)
    t1=time.time()
    samples,weights,names,logz=stage.samples,stage.weights,stage.param_names,stage.log_z
    _print_summary("stage-L",samples,weights,names); print(f"[stage-L] time taken: {t1-t0:.2f} seconds")
    medians=stage.medians(); q50=[medians[n] for n in names]
    likelihood.set_values(q50)
    fwd=likelihood.forward_model(use_linear=True,return_intensity=True,ret_each_plane=True,
                                  image_map=likelihood.image_data,noise_map=likelihood.noise_map)
    lens_image_model=np.asarray(fwd[1])
    _dump_stage("l",samples,weights,names,logz,
                extra=dict(medians=medians,lens_light_model=lens_image_model,time_taken=t1-t0),stage=stage)
    try:
        plot_model_results(likelihood,jnp.asarray(q50),save_path=str(OUT_DIR/"stage_l_model.png"),
                          title="Stage L : arc-masked Bspline lens light")
    except Exception as err: print(f"[stage-L] plotting failed (non-fatal): {err}")
    try:
        llt=np.asarray(fits.getdata(DATA_DIR/"lens_light_true.fits")); diff=llt-lens_image_model; dn=diff/noise_map
        npix=image_data.shape[0]; ext=[-npix*DPIX/2,npix*DPIX/2,-npix*DPIX/2,npix*DPIX/2]
        fig,axes=plt.subplots(1,4,figsize=(18,4.2))
        for ax,img,title,kw in [
            (axes[0],llt,"True lens light (X)",dict(vmin=0,cmap="viridis")),
            (axes[1],lens_image_model,"Fitted lens light (M)",dict(vmin=0,cmap="viridis")),
            (axes[2],diff,"Residual (X - M)",dict(cmap="RdBu_r")),
            (axes[3],dn,"(X - M) / noise",dict(vmin=-3,vmax=3,cmap="RdBu_r")),
        ]:
            im=ax.imshow(img,origin="lower",extent=ext,**kw); ax.set_title(title,fontsize=11)
            ax.set_xlabel("arcsec"); plt.colorbar(im,ax=ax,fraction=0.046,pad=0.04)
        axes[0].set_ylabel("arcsec"); plt.suptitle("Stage L : lens light diagnostic",fontsize=11)
        plt.tight_layout(); plt.savefig(OUT_DIR/"stage_l_diagnostic.png",dpi=120,bbox_inches="tight"); plt.close(fig)
        print(f"[stage-L] diagnostic plot saved to {OUT_DIR/'stage_l_diagnostic.png'}")
    except Exception as err: print(f"[stage-L] diagnostic plot failed (non-fatal): {err}")
    return stage,medians,lens_image_model


# ------------------------------------------------------------------ #
# Position likelihood
# ------------------------------------------------------------------ #
def _position_likelihood_from_stage_a(medians_a):
    required=("theta_E","e1_mass","e2_mass","center_x_mass","center_y_mass","gamma1","gamma2","center_x_src","center_y_src")
    missing=[k for k in required if k not in medians_a]
    if missing: raise KeyError("Stage-A medians missing: "+", ".join(missing))
    mass=PhysicalModel(lens_mass=[SIE(theta_E=float(medians_a["theta_E"]),e1=float(medians_a["e1_mass"]),
                                       e2=float(medians_a["e2_mass"]),center_x=float(medians_a["center_x_mass"]),
                                       center_y=float(medians_a["center_y_mass"])),
                                   Shear(gamma1=float(medians_a["gamma1"]),gamma2=float(medians_a["gamma2"]))],
                        source_light=[],lens_light=[])
    solver=PointSourceProbModel(phys_model=mass,observed_positions=[[0.0,0.0]],position_sigma=[0.01],
                                 source_x=float(medians_a["center_x_src"]),source_y=float(medians_a["center_y_src"]),
                                 source_position_fixed=True,solver="optimization",
                                 solver_config={"initial_range":3.0,"n_x":200,"n_y":200,"k_keep":30,"num_iters":20,"tolerance":5.0e-4,"cluster_tol":0.08})
    ip,_=solver.solve_image_positions(); ip=np.asarray(ip,dtype=np.float64)
    if ip.ndim!=2 or ip.shape[1]!=2: raise RuntimeError(f"Invalid image_positions shape={ip.shape}")
    if ip.shape[0]<2: raise RuntimeError("Fewer than 2 lensed image positions")
    print(f"[pos-like] solved {ip.shape[0]} lensed image positions from stage-A medians")
    for p in ip: print(f"    ({p[0]:+.4f}, {p[1]:+.4f})")
    return dict(positions=ip.tolist(),threshold_arcsec=0.3,min_log_like=-1.0e10)


# ------------------------------------------------------------------ #
# Mass helpers
# ------------------------------------------------------------------ #
def _sie_mass_from_stage_a(stage_a):
    return (SIE(theta_E=stage_a.fixed("theta_E"),e1=stage_a.fixed("e1_mass"),e2=stage_a.fixed("e2_mass"),
                center_x=stage_a.fixed("center_x_mass"),center_y=stage_a.fixed("center_y_mass")),
            Shear(gamma1=stage_a.fixed("gamma1"),gamma2=stage_a.fixed("gamma2")))


def _epl_mass_from_stage(stage):
    te=stage.gaussian("theta_E",model="EPL",attr="theta_E",limits=[0.0,5.0])
    e1m=stage.gaussian("e1_mass",model="EPL",attr="e1",limits=[-1.0,1.0])
    e2m=stage.gaussian("e2_mass",model="EPL",attr="e2",limits=[-1.0,1.0])
    cx=stage.gaussian("center_x_mass",model="EPL",attr="center_x",limits=[-1.0,1.0])
    cy=stage.gaussian("center_y_mass",model="EPL",attr="center_y",limits=[-1.0,1.0])
    gamma=ParamU("gamma",2.0,prior_type="uniform",prior_settings=[1.0,3.0],limits=[1.0,3.0])
    epl=EPL(theta_E=te,gamma=gamma,e1=e1m,e2=e2m,center_x=cx,center_y=cy); epl.gamma.to_dynamic()
    return epl,Shear(gamma1=stage.gaussian("gamma1",model="Shear",attr="gamma1",limits=[-0.5,0.5]),
                      gamma2=stage.gaussian("gamma2",model="Shear",attr="gamma2",limits=[-0.5,0.5]))


def _epl_mass_from_medians(medians):
    required=("theta_E","gamma","e1_mass","e2_mass","center_x_mass","center_y_mass","gamma1","gamma2")
    missing=[n for n in required if n not in medians]
    if missing: raise KeyError("EPL medians missing: "+", ".join(missing))
    epl=EPL(theta_E=ParamU("theta_E",float(medians["theta_E"])),gamma=ParamU("gamma",float(medians["gamma"])),
            e1=ParamU("e1_mass",float(medians["e1_mass"])),e2=ParamU("e2_mass",float(medians["e2_mass"])),
            center_x=ParamU("center_x_mass",float(medians["center_x_mass"])),center_y=ParamU("center_y_mass",float(medians["center_y_mass"])))
    for p in (epl.theta_E,epl.gamma,epl.e1,epl.e2,epl.center_x,epl.center_y): p.to_static()
    shear=Shear(gamma1=ParamU("gamma1",float(medians["gamma1"])),gamma2=ParamU("gamma2",float(medians["gamma2"])))
    shear.gamma1.to_static(); shear.gamma2.to_static()
    return epl,shear


# ------------------------------------------------------------------ #
# _plot_pix_stage
# ------------------------------------------------------------------ #
def _plot_pix_stage(tag,likelihood,medians,param_names,save_path):
    q50=[medians[n] for n in param_names]
    idata=np.asarray(likelihood.image_data); nmap=np.asarray(likelihood.noise_map)
    mask=~np.asarray(likelihood.unmask,dtype=bool); hpp=False
    with ck.ActiveContext(likelihood):
        likelihood.fill_params(jnp.array(q50))
        lv=jnp.exp(likelihood.phys_model.source_light[0].log_lambda_reg.value); lj=jnp.asarray(lv)
        rv=float(np.asarray(_source_param_value(likelihood.phys_model.source_light[0].adaptive_reg_rho)))
        (xmin,xmax,ymin,ymax,bxs,bys,bx_seed,by_seed)=likelihood._get_bbox()
        scale=likelihood._get_reg_scale()
        rd=likelihood._regularization_data(xmin,xmax,ymin,ymax,scale=scale)
        od=likelihood.sim_obj.precompute_operator_data(xmin,xmax,ymin,ymax,_betas_sub=(bxs,bys))
        bc,bm=likelihood.sim_obj.build_block_diag_preconditioner(
            likelihood.noise_1d,xmin,xmax,ymin,ymax,lj,likelihood.reg_builder,block_size=likelihood.block_size,scale=scale)
        sp,_=likelihood._solve_source(xmin,xmax,ymin,ymax,lj,rd,(bc,bm),op_data=od)
        m1d=likelihood.sim_obj.forward_model(sp,xmin,xmax,ymin,ymax,op_data=od)
        ns=likelihood.sim_obj.source_n; bs=likelihood.block_size; nb=(ns+bs-1)//bs
        tipr=jnp.array(0.0,dtype=lj.dtype)
        for by in range(nb):
            for bx in range(nb):
                bid=bx+by*nb; xs,xe=bx*bs,min((bx+1)*bs,ns); ys,ye=by*bs,min((by+1)*bs,ns)
                if bid>=len(bc): break
                Rb=likelihood.reg_builder.block_diag_R(xs,xe,ys,ye,xmin,xmax,ymin,ymax,scale=scale)
                tipr=tipr+jnp.trace(jsl.cho_solve((bc[bid],True),Rb))
        Neff=float(likelihood.sim_obj.n_source_pixels-lj*tipr)
        hpp=likelihood._has_pos_penalty
        if hpp:
            ppen=float(likelihood._position_likelihood_penalty_jax())
            bX,bY=likelihood.phys_model.deflection(likelihood._pos_px,likelihood._pos_py)
            dx=bX[:,None]-bX[None,:]; dy=bY[:,None]-bY[None,:]; ms=float(jnp.max(jnp.sqrt(dx*dx+dy*dy)))
            print(f"[{tag}] Position likelihood penalty: {ppen:.4e}")
            print(f"[{tag}] Maximum source-plane separation of marked images: {ms:.4e} arcsec")
            bXn=np.array(bX); bYn=np.array(bY); pxn=np.array(likelihood._pos_px); pyn=np.array(likelihood._pos_py)
    m1d_np=np.array(m1d); mi=np.zeros(idata.shape); mi[~mask]=m1d_np; rn=(idata-mi)/nmap
    chi2=float(np.sum(rn[~mask]**2)); dof=int((~mask).sum())-Neff; chi2_nu=chi2/dof if dof>0 else 0.0
    n=likelihood.phys_model.source_light[0].n; si=np.array(sp).reshape(n,n)
    sci=np.array(scale).reshape(n,n) if scale is not None else np.ones((n,n))
    npix=idata.shape[0]
    ext_i=[-npix*DPIX/2,npix*DPIX/2,-npix*DPIX/2,npix*DPIX/2]
    ext_s=[float(xmin),float(xmax),float(ymin),float(ymax)]
    vmax=np.nanpercentile(idata[~mask],99.5)
    fig,axes=plt.subplots(1,5,figsize=(21,4.2))
    ru,cu=np.where(~mask); pad=3
    rmin=max(ru.min()-pad,0); rmax=min(ru.max()+pad,npix-1); cmin=max(cu.min()-pad,0); cmax=min(cu.max()+pad,npix-1)
    xl=(-npix*DPIX/2+cmin*DPIX,-npix*DPIX/2+(cmax+1)*DPIX); yl=(-npix*DPIX/2+rmin*DPIX,-npix*DPIX/2+(rmax+1)*DPIX)
    for ax,img,title,kw in [
        (axes[0],idata,"Data (lens-subtracted)",dict(vmin=0,vmax=vmax,cmap="viridis")),
        (axes[1],mi,"Model image",dict(vmin=0,vmax=vmax,cmap="viridis")),
        (axes[2],np.where(mask,np.nan,rn),f"Norm. residual\nχ²/ν={chi2_nu:.3f}",dict(vmin=-3,vmax=3,cmap="RdBu_r")),
    ]:
        im=ax.imshow(img,origin="lower",extent=ext_i,**kw)
        if ax==axes[0] and hpp: ax.plot(pxn,pyn,'rx',markersize=8,label='Marked pos'); ax.legend(loc='upper right',fontsize=8)
        ax.set_title(title,fontsize=11); ax.set_xlabel("arcsec"); plt.colorbar(im,ax=ax,fraction=0.046,pad=0.04)
    im3=axes[3].imshow(si,origin="lower",extent=ext_s,cmap="viridis")
    if hpp: axes[3].plot(bXn,bYn,'rx',markersize=8,label='Traced pos'); axes[3].legend(loc='upper right',fontsize=8)
    axes[3].set_title(f"Source reconstruction\n(λ={float(lv):.2e})",fontsize=11); axes[3].set_xlabel("arcsec")
    plt.colorbar(im3,ax=axes[3],fraction=0.046,pad=0.04)
    im4=axes[4].imshow(sci,origin="lower",extent=ext_s,cmap="plasma",vmin=1.0,vmax=max(1.0,float(np.nanmax(sci))))
    axes[4].set_title(f"Reg precision scale\n(rho={rv:.2f})",fontsize=11); axes[4].set_xlabel("arcsec")
    plt.colorbar(im4,ax=axes[4],fraction=0.046,pad=0.04,label=r"$\lambda_i / \lambda_{\rm global}$")
    axes[0].set_ylabel("arcsec")
    for ax in axes[:3]: ax.set_xlim(*xl); ax.set_ylim(*yl)
    lbl="  ".join(f"{n}={medians[n]:+.4f}" for n in ("theta_E","gamma","e1_mass","e2_mass") if n in medians)
    plt.suptitle(f"[{tag}]  {lbl}",fontsize=10)
    overlay_critical_and_caustics(image_axes=[axes[0],axes[1]],source_ax=axes[3],lens_mass=likelihood.phys_model)
    plt.tight_layout(); plt.savefig(save_path,dpi=120,bbox_inches="tight"); plt.close(fig)
    print(f"[{tag}] diagnostic plot saved to {save_path}")


# ------------------------------------------------------------------ #
# Stage M0-M3
# ------------------------------------------------------------------ #
def build_stage_m0_likelihood(ls_img,nmap,psk,fmask,stage_a,pl,cmask=None):
    sie,shear=_sie_mass_from_stage_a(stage_a)
    ll=ParamU("log_lambda_reg",0.0,prior_type="uniform",prior_settings=[-13.815510557964274,13.815510557964274],limits=[-13.815510557964274,13.815510557964274]); ll.to_dynamic()
    phys=PhysicalModel(lens_mass=[sie,shear],source_light=[PixelizedSourceModel(n=NSRC,log_lambda_reg=ll,regularization_type=PIXEL_REGULARIZATION_TYPE)],lens_light=[])
    cm=fmask;
    if cmask is not None: cm=cm|cmask
    return PixelizedImageProbModelOperator(image_data=ls_img,noise_map=nmap,psf_kernel=psk,dpix=DPIX,nsub=NSUB_PIX,
                                            phys_model=phys,mask=cm,position_likelihood=pl,solver_type=SOLVER_TYPE,
                                            source_bbox_padding=SOURCE_BBOX_PADDING,**_fista_kwargs())

def run_stage_m0(idata,nmap,psk,fmask,llm,stage_a,pl,cmask=None):
    print("\n"+"="*60); print(" Stage M0 : fixed SIE + shear + uniform pix source (build S0)"); print("="*60)
    t0=time.time(); ls=idata-llm
    lkl=build_stage_m0_likelihood(ls,nmap,psk,fmask,stage_a,pl,cmask)
    llb=make_likelihood(lkl,vectorized=True); ng=200
    lmin,lmax=jnp.log(1e-8),jnp.log(1e8)
    lgc=jnp.linspace(lmin,lmax,ng); lec=jnp.asarray(llb(lgc.reshape(-1,1)))
    vc=_valid_log_evidence(lec)
    if not np.any(vc): raise RuntimeError("[stage-M0] All coarse grid values failed.")
    bi=int(np.argmax(np.where(vc,np.asarray(lec),-np.inf))); lbc=float(lgc[bi])
    print(f"[stage-M0] Coarse best: λ = {float(jnp.exp(lbc)):.4e}")
    hw=0.5*jnp.log(10); lgf=jnp.linspace(lbc-hw,lbc+hw,ng); lef=jnp.asarray(llb(lgf.reshape(-1,1)))
    vf=_valid_log_evidence(lef)
    if not np.any(vf): raise RuntimeError("[stage-M0] All refinement grid values failed.")
    bif=int(np.argmax(np.where(vf,np.asarray(lef),-np.inf))); llb_f=float(lgf[bif]); leb=float(lef[bif])
    print(f"[stage-M0] Refined best: λ = {float(jnp.exp(llb_f)):.4e}")
    ma=stage_a.medians()
    s0=_solve_pixel_source_for_package(lkl,{**ma,"log_lambda_reg":llb_f},["log_lambda_reg"])
    s0.update(lambda_best=float(jnp.exp(llb_f)),log_lambda_best=llb_f,evidence_lambda_best=float(jnp.exp(llb_f)),
              evidence_log_lambda_best=llb_f,stage_a_medians=dict(ma))
    s0["scale_map"]=np.asarray(_make_s0_scale(s0),dtype=np.float32); _validate_s0_package(s0)
    t1=time.time(); print(f"[stage-M0] time taken: {t1-t0:.2f} seconds")
    _dump_stage("m0",None,None,["log_lambda_reg"],leb,
                extra=dict(log_lambda_best=llb_f,evidence_lambda_best=float(jnp.exp(llb_f)),evidence_log_lambda_best=llb_f,
                           s0=s0,time_taken=t1-t0,
                           lambda_grid_coarse=np.asarray(lgc,dtype=np.float64),log_ev_coarse=np.asarray(lec,dtype=np.float64),
                           lambda_grid_fine=np.asarray(lgf,dtype=np.float64),log_ev_fine=np.asarray(lef,dtype=np.float64)))
    try: _plot_pix_stage("stage-M0",lkl,{**ma,"log_lambda_reg":llb_f},["log_lambda_reg"],str(OUT_DIR/"stage_m0_model.png"))
    except Exception as err: print(f"[stage-M0] plot failed: {err}")
    return s0,llb_f

def build_stage_m1_likelihood(ls_img,nmap,psk,fmask,stage_a,pl,ll_fixed,cmask=None):
    epl,shear=_epl_mass_from_stage(stage_a)
    ll=ParamU("log_lambda_reg",float(ll_fixed),prior_type="truncated_gaussian",
              prior_settings=[float(ll_fixed),0.15],limits=[float(ll_fixed)-0.5,float(ll_fixed)+0.5]); ll.to_dynamic()
    phys=PhysicalModel(lens_mass=[epl,shear],source_light=[PixelizedSourceModel(n=NSRC,log_lambda_reg=ll,regularization_type=PIXEL_REGULARIZATION_TYPE)],lens_light=[])
    cm=fmask;
    if cmask is not None: cm=cm|cmask
    return PixelizedImageProbModelOperator(image_data=ls_img,noise_map=nmap,psf_kernel=psk,dpix=DPIX,nsub=NSUB_PIX,
                                            phys_model=phys,mask=cm,position_likelihood=pl,solver_type=SOLVER_TYPE,
                                            source_bbox_padding=SOURCE_BBOX_PADDING,**_fista_kwargs())

def run_stage_m1(idata,nmap,psk,fmask,llm,stage_a,pl,ll_fixed,cmask=None):
    print("\n"+"="*60); print(" Stage M1 : EPL + shear + non-adaptive pix source (mass fit)"); print("="*60)
    t0=time.time(); ls=idata-llm
    lkl=build_stage_m1_likelihood(ls,nmap,psk,fmask,stage_a,pl,ll_fixed,cmask)
    stage=_run_sampler(lkl,n_live=300,n_eff=600,tag="stage-M1",vectorized=True)
    t1=time.time(); s,w,n,lz=stage.samples,stage.weights,stage.param_names,stage.log_z
    _print_summary("stage-M1",s,w,n); print(f"[stage-M1] time taken: {t1-t0:.2f} seconds")
    med=stage.medians(); sm=dict(med); llm1=float(med["log_lambda_reg"])
    s1=_solve_pixel_source_for_package(lkl,sm,n)
    s1.update(lambda_best=float(jnp.exp(llm1)),log_lambda_best=llm1,stage_m1_medians=dict(med))
    s1["scale_map"]=np.asarray(_make_s0_scale(s1),dtype=np.float32); _validate_s0_package(s1)
    _dump_stage("m1",s,w,n,lz,extra=dict(medians=med,m1_mass_model="EPL+Shear",
                lambda_prior_center=float(ll_fixed),lambda_prior_sigma=0.15,
                lambda_prior_limits=[float(ll_fixed)-0.5,float(ll_fixed)+0.5],
                s1=s1,time_taken=t1-t0),stage=stage)
    try: _plot_pix_stage("stage-M1",lkl,med,n,str(OUT_DIR/"stage_m1_model.png"))
    except Exception as err: print(f"[stage-M1] plot failed: {err}")
    return stage,med,s1

def build_stage_m2_likelihood(ls_img,nmap,psk,fmask,med_m1,pl,s1_pkg,cmask=None):
    epl,shear=_epl_mass_from_medians(med_m1)
    ll=ParamU("log_lambda_reg",float(s1_pkg["log_lambda_best"]),prior_type="uniform",
              prior_settings=[-13.815510557964274,13.815510557964274],limits=[-13.815510557964274,13.815510557964274]); ll.to_dynamic()
    rho=ParamU("adaptive_reg_rho",ADAPTIVE_REG_RHO,prior_type="uniform",
               prior_settings=[0.0,ADAPTIVE_REG_RHO_PRIOR_MAX],limits=[0.0,ADAPTIVE_REG_RHO_PRIOR_MAX]); rho.to_dynamic()
    phys=PhysicalModel(lens_mass=[epl,shear],
                       source_light=[PixelizedSourceModel(n=NSRC,log_lambda_reg=ll,regularization_type=PIXEL_REGULARIZATION_TYPE,adaptive_reg_rho=rho)],
                       lens_light=[])
    cm=fmask;
    if cmask is not None: cm=cm|cmask
    return PixelizedImageProbModelOperator(image_data=ls_img,noise_map=nmap,psf_kernel=psk,dpix=DPIX,nsub=NSUB_PIX,
                                            phys_model=phys,mask=cm,position_likelihood=pl,solver_type=SOLVER_TYPE,
                                            source_bbox_padding=SOURCE_BBOX_PADDING,**_fista_kwargs(),**_s0_fixed_kwargs(s1_pkg))

def run_stage_m2(idata,nmap,psk,fmask,llm,med_m1,pl,s1_pkg,cmask=None):
    print("\n"+"="*60); print(" Stage M2 : fixed EPL + shear + adaptive pix source (fit λ, rho)"); print("="*60)
    t0=time.time(); ls=idata-llm
    lkl=build_stage_m2_likelihood(ls,nmap,psk,fmask,med_m1,pl,s1_pkg,cmask)
    stage=_run_sampler(lkl,n_live=250,n_eff=600,tag="stage-M2",vectorized=True)
    t1=time.time(); s,w,n,lz=stage.samples,stage.weights,stage.param_names,stage.log_z
    _print_summary("stage-M2",s,w,n); print(f"[stage-M2] time taken: {t1-t0:.2f} seconds")
    med=stage.medians(); rhm2={"log_lambda_reg":float(med["log_lambda_reg"]),"adaptive_reg_rho":float(med["adaptive_reg_rho"])}
    print(f"[stage-M2] median source reg: {_format_reg_hyperparams(rhm2)}")
    _dump_stage("m2",s,w,n,lz,extra=dict(medians=med,m2_mass_model="fixed-EPL+Shear",mass_medians_fixed=dict(med_m1),
                reg_hyperparams=rhm2,time_taken=t1-t0),stage=stage)
    try: _plot_pix_stage("stage-M2",lkl,med,n,str(OUT_DIR/"stage_m2_model.png"))
    except Exception as err: print(f"[stage-M2] plot failed: {err}")
    return stage,med,rhm2

def build_stage_m3_likelihood(ls_img,nmap,psk,fmask,stage_m1,pl,rh_fixed,s1_pkg,cmask=None):
    epl,shear=_epl_mass_from_stage(stage_m1)
    ll=ParamU("log_lambda_reg",float(rh_fixed["log_lambda_reg"])); ll.to_static()
    rho=ParamU("adaptive_reg_rho",float(rh_fixed["adaptive_reg_rho"])); rho.to_static()
    phys=PhysicalModel(lens_mass=[epl,shear],
                       source_light=[PixelizedSourceModel(n=NSRC,log_lambda_reg=ll,regularization_type=PIXEL_REGULARIZATION_TYPE,adaptive_reg_rho=rho)],
                       lens_light=[])
    cm=fmask;
    if cmask is not None: cm=cm|cmask
    return PixelizedImageProbModelOperator(image_data=ls_img,noise_map=nmap,psf_kernel=psk,dpix=DPIX,nsub=NSUB_PIX,
                                            phys_model=phys,mask=cm,position_likelihood=pl,solver_type=SOLVER_TYPE,
                                            source_bbox_padding=SOURCE_BBOX_PADDING,**_fista_kwargs(),**_s0_fixed_kwargs(s1_pkg))

def run_stage_m3(idata,nmap,psk,fmask,llm,stage_m1,pl,rh_fixed,s1_pkg,cmask=None):
    print("\n"+"="*60); print(" Stage M3 : EPL + shear + adaptive pix source (final mass fit)"); print("="*60)
    print(f"[stage-M3] fixed source reg: {_format_reg_hyperparams(rh_fixed)}")
    t0=time.time(); ls=idata-llm
    lkl=build_stage_m3_likelihood(ls,nmap,psk,fmask,stage_m1,pl,rh_fixed,s1_pkg,cmask)
    stage=_run_sampler(lkl,n_live=300,n_eff=600,tag="stage-M3",vectorized=True)
    t1=time.time(); s,w,n,lz=stage.samples,stage.weights,stage.param_names,stage.log_z
    _print_summary("stage-M3",s,w,n); print(f"[stage-M3] time taken: {t1-t0:.2f} seconds")
    med=stage.medians()
    _dump_stage("m3",s,w,n,lz,extra=dict(medians=med,m3_mass_model="EPL+Shear",reg_hyperparams_fixed=dict(rh_fixed),time_taken=t1-t0),stage=stage)
    try: _plot_pix_stage("stage-M3",lkl,med,n,str(OUT_DIR/"stage_m3_model.png"))
    except Exception as err: print(f"[stage-M3] plot failed: {err}")
    return stage,med


# ------------------------------------------------------------------ #
# Main
# ------------------------------------------------------------------ #
def main(skip_done=False, out_dir=None):
    global OUT_DIR
    if out_dir is not None: OUT_DIR=Path(out_dir)
    idata,nmap,psk,_=load_lens_data(image_path=str(DATA_DIR/"image.fits"),noise_path=str(DATA_DIR/"noise.fits"),psf_path=str(DATA_DIR/"psf.fits"))
    cmask=_make_circular_mask(idata.shape,DPIX,radius_arcsec=MASK_RADIUS)
    print(f"[circular mask] excluded {int(cmask.sum())} / {cmask.size} pixels (> {MASK_RADIUS} arcsec)")

    ta=0.0
    if skip_done and (OUT_DIR/"stage_a.pkl").exists():
        print(f"[stage-A] loading cached {OUT_DIR}/stage_a.pkl"); d=_load_stage("a")
        stage_a=_stage_from_payload(d); ma=d["extra"]["medians"]; llm=d["extra"]["lens_light_model"]; ta=d["extra"].get("time_taken",0.0)
    else: stage_a,ma,llm=run_stage_a(idata,nmap,psk,cmask); ta=_load_stage("a")["extra"].get("time_taken",0.0)

    fmask=run_stage_b(idata,nmap,llm,cmask)

    tl=0.0
    if skip_done and (OUT_DIR/"stage_l.pkl").exists():
        print(f"[stage-L] loading cached {OUT_DIR}/stage_l.pkl"); d=_load_stage("l")
        stage_l=_stage_from_payload(d); ml=d["extra"]["medians"]; llm=d["extra"]["lens_light_model"]; tl=d["extra"].get("time_taken",0.0)
    else: stage_l,ml,llm=run_stage_l(idata,nmap,psk,fmask,stage_a,cmask); tl=_load_stage("l")["extra"].get("time_taken",0.0)

    ls=idata-llm; pl=_position_likelihood_from_stage_a(ma)

    tm0=0.0
    if skip_done and (OUT_DIR/"stage_m0.pkl").exists():
        print(f"[stage-M0] loading cached {OUT_DIR}/stage_m0.pkl"); d=_load_stage("m0")
        s0=_validate_s0_package(d["extra"]["s0"]); llm0=d["extra"]["log_lambda_best"]; tm0=d["extra"].get("time_taken",0.0)
    else: s0,llm0=run_stage_m0(idata,nmap,psk,fmask,llm,stage_a,pl,cmask); tm0=_load_stage("m0")["extra"].get("time_taken",0.0)
    if (OUT_DIR/"stage_m0.pkl").exists() and not (OUT_DIR/"stage_m0_model.png").exists():
        lkl=build_stage_m0_likelihood(ls,nmap,psk,fmask,stage_a,pl,cmask)
        try: _plot_pix_stage("stage-M0",lkl,{**ma,"log_lambda_reg":float(s0["log_lambda_best"])},["log_lambda_reg"],str(OUT_DIR/"stage_m0_model.png"))
        except Exception as err: print(f"[stage-M0] re-plot failed: {err}")

    tm1=0.0; s1=None
    if skip_done and (OUT_DIR/"stage_m1.pkl").exists():
        print(f"[stage-M1] loading cached {OUT_DIR}/stage_m1.pkl"); d=_load_stage("m1")
        try: nm1=d["param_names"]; stage_m1=_stage_from_payload(d); mm1=d["extra"]["medians"]; s1=_validate_s0_package(d["extra"]["s1"]); tm1=d["extra"].get("time_taken",0.0)
        except KeyError as err:
            print(f"[stage-M1] old format ({err}); recomputing.")
            stage_m1,mm1,s1=run_stage_m1(idata,nmap,psk,fmask,llm,stage_a,pl,llm0,cmask); nm1=stage_m1.param_names; tm1=_load_stage("m1")["extra"].get("time_taken",0.0)
    else: stage_m1,mm1,s1=run_stage_m1(idata,nmap,psk,fmask,llm,stage_a,pl,llm0,cmask); nm1=stage_m1.param_names; tm1=_load_stage("m1")["extra"].get("time_taken",0.0)
    if s1 is None: raise RuntimeError("[stage-M1] Failed to determine S1 source package.")
    if (OUT_DIR/"stage_m1.pkl").exists() and not (OUT_DIR/"stage_m1_model.png").exists():
        lkl=build_stage_m1_likelihood(ls,nmap,psk,fmask,stage_a,pl,llm0,cmask)
        try: _plot_pix_stage("stage-M1",lkl,mm1,nm1,str(OUT_DIR/"stage_m1_model.png"))
        except Exception as err: print(f"[stage-M1] re-plot failed: {err}")

    tm2=0.0; rhm2=None
    if skip_done and (OUT_DIR/"stage_m2.pkl").exists():
        print(f"[stage-M2] loading cached {OUT_DIR}/stage_m2.pkl"); d=_load_stage("m2")
        stage_m2=_stage_from_payload(d); nm2=stage_m2.param_names; mm2=d["extra"]["medians"]; rhm2=_reg_hyperparams_from_m2_payload(d); tm2=d["extra"].get("time_taken",0.0)
    else: stage_m2,mm2,rhm2=run_stage_m2(idata,nmap,psk,fmask,llm,mm1,pl,s1,cmask); nm2=stage_m2.param_names; tm2=_load_stage("m2")["extra"].get("time_taken",0.0)
    if not (OUT_DIR/"stage_m2_model.png").exists():
        lkl=build_stage_m2_likelihood(ls,nmap,psk,fmask,mm1,pl,s1,cmask)
        try: _plot_pix_stage("stage-M2",lkl,mm2,nm2,str(OUT_DIR/"stage_m2_model.png"))
        except Exception as err: print(f"[stage-M2] plot failed: {err}")
    if rhm2 is None: raise RuntimeError("[stage-M2] Failed to determine source regularization hyperparameters.")

    tm3=0.0
    if skip_done and (OUT_DIR/"stage_m3.pkl").exists():
        print(f"[stage-M3] loading cached {OUT_DIR}/stage_m3.pkl"); d=_load_stage("m3")
        stage_m3=_stage_from_payload(d); nm3=stage_m3.param_names; mm3=d["extra"]["medians"]; tm3=d["extra"].get("time_taken",0.0)
    else: stage_m3,mm3=run_stage_m3(idata,nmap,psk,fmask,llm,stage_m1,pl,rhm2,s1,cmask); nm3=stage_m3.param_names; tm3=_load_stage("m3")["extra"].get("time_taken",0.0)
    if not (OUT_DIR/"stage_m3_model.png").exists():
        lkl=build_stage_m3_likelihood(ls,nmap,psk,fmask,stage_m1,pl,rhm2,s1,cmask)
        try: _plot_pix_stage("stage-M3",lkl,mm3,nm3,str(OUT_DIR/"stage_m3_model.png"))
        except Exception as err: print(f"[stage-M3] plot failed: {err}")

    print("\n"+"="*60); print(" Pipeline complete"); print("="*60)
    print(" Time summary:"); print(f"    Stage A:  {ta/60:.2f} min"); print(f"    Stage L:  {tl/60:.2f} min")
    print(f"    Stage M0: {tm0/60:.2f} min"); print(f"    Stage M1: {tm1/60:.2f} min")
    print(f"    Stage M2: {tm2/60:.2f} min"); print(f"    Stage M3: {tm3/60:.2f} min")
    print(f"    Total:    {(ta+tl+tm0+tm1+tm2+tm3)/60:.2f} min\n")
    print(f"    M0 best lambda_reg     = {float(jnp.exp(llm0)):.4e}")
    print(f"    M1 median gamma        = {mm1.get('gamma',np.nan):+.4f}")
    print(f"    M2 median source reg   = {_format_reg_hyperparams(rhm2)}")
    for k in ("theta_E","gamma","e1_mass","e2_mass","center_x_mass","center_y_mass","gamma1","gamma2"):
        if k in mm3: print(f"    final  {k:15s} = {mm3[k]:+.4f}")

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--skip-done",action="store_true"); parser.add_argument("--out-dir",default=None)
    args=parser.parse_args(); main(skip_done=args.skip_done,out_dir=args.out_dir)
