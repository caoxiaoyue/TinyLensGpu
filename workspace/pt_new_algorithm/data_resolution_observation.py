"""Exact block averaging after the existing fine-grid lens/PSF operator.

No effective coarse PSF approximation, resampled potential grid, or artificial
noise reset is introduced. Only complete, disjoint blocks in the original
likelihood mask are used; their propagated Gaussian covariance is diagonal.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from TinyLensGpu.ObservationModel import IterativePotentialCorrection
from TinyLensGpu.ObservationModel.LensImage.potential_correction_iterative import _CurrentPoint


class ObservationProjection:
    def __init__(self, fine_mask, factor=1, *, matched_to=None):
        mask=np.asarray(fine_mask,dtype=bool)
        if mask.ndim!=2 or mask.shape[0]!=mask.shape[1]:
            raise ValueError("Use a square native image mask")
        if isinstance(factor,bool) or int(factor)!=factor or factor<1:
            raise ValueError("factor must be a positive integer")
        self.factor=int(factor)
        self.fine_shape=mask.shape
        self.fine_flat=np.flatnonzero(~mask.ravel())
        self.n_fine=len(self.fine_flat)
        lookup=np.full(mask.size,-1,dtype=np.int32)
        lookup[self.fine_flat]=np.arange(self.n_fine)
        if matched_to is not None:
            if self.factor!=1 or not np.array_equal(self.fine_flat,matched_to.fine_flat):
                raise ValueError("A matched unbinned view needs factor 1 and the same fine mask")
            selected=np.unique(np.asarray(matched_to.indices).ravel())
            self.shape=mask.shape
            self.crop_start=0
            flat=self.fine_flat[selected]
            groups=selected[:,None]
        else:
            side=mask.shape[0]//self.factor
            width=side*self.factor
            self.crop_start=(mask.shape[0]-width)//2
            self.shape=(side,side)
            ids=lookup.reshape(mask.shape)[self.crop_start:self.crop_start+width,
                                           self.crop_start:self.crop_start+width]
            blocks=ids.reshape(side,self.factor,side,self.factor).transpose(0,2,1,3)
            groups_all=blocks.reshape(side*side,self.factor*self.factor)
            valid=np.all(groups_all>=0,axis=1)
            flat=np.flatnonzero(valid)
            groups=groups_all[valid]
        if not len(groups):
            raise ValueError("No complete blocks remain")
        self.indices=jnp.asarray(groups,dtype=jnp.int32)
        self.flat_indices=jnp.asarray(flat,dtype=jnp.int32)
        self.n_data=len(flat)
        self.group_size=groups.shape[1]
        self.mask=np.ones(self.shape,dtype=bool)
        self.mask.ravel()[flat]=False
        if len(np.unique(groups))!=groups.size:
            raise ValueError("Disjoint blocks are required for diagonal propagated noise")

    def apply(self, fine_values):
        return jnp.mean(fine_values[self.indices],axis=1)

    def adjoint(self, values):
        weights=jnp.broadcast_to(values[:,None]/self.group_size,self.indices.shape)
        return jnp.zeros(self.n_fine,dtype=values.dtype).at[self.indices.ravel()].add(weights.ravel())

    def noise(self, fine_sigma):
        return jnp.sqrt(jnp.sum(fine_sigma[self.indices]**2,axis=1))/self.group_size

    def full(self, active):
        return jnp.full(int(np.prod(self.shape)),jnp.nan,dtype=active.dtype).at[self.flat_indices].set(active).reshape(self.shape)


class ProjectedPotentialCorrection(IterativePotentialCorrection):
    """Same physical lens/source unknowns with a projected observation vector."""

    def __init__(self, *args, projection, **kwargs):
        super().__init__(*args,**kwargs)
        if not np.array_equal(np.asarray(self.renderer.flat_indices),projection.fine_flat):
            raise ValueError("Projection must use the actual fine renderer mask")
        self.projection=projection
        self.fine_data_active=self.data_active
        self.fine_noise_active=self.noise_active
        self.data_active=projection.apply(self.fine_data_active)
        self.noise_active=projection.noise(self.fine_noise_active)
        self.noise_logdet=2*jnp.log(self.noise_active).sum()
        self.initialization_provenance.update(observation_operator="block_mean_after_fine_PSF",
            fine_geometry_unchanged=True, extra_fitted_parameters=0)

    def source_forward(self, source, mapping=None):
        return self.projection.apply(self.renderer.source_forward(source,mapping))

    def forward(self, joint_values):
        return self.projection.apply(super().forward(joint_values))

    def jacobian_forward(self, increment, linearization):
        return self.projection.apply(super().jacobian_forward(increment,linearization))

    def jacobian_adjoint(self, values, linearization):
        return super().jacobian_adjoint(self.projection.adjoint(values),linearization)

    def _evaluate(self, values, lambda_source, lambda_potential):
        linearization=self.linearize(values)
        image=self.source_forward(values[:self.n_source],linearization.mapping)
        residual=self.data_active-image
        prior=self.prior_matvec(values,lambda_source,lambda_potential)
        negative=self.jacobian_adjoint(residual/self.noise_active**2,linearization)-prior
        chi2=jnp.vdot(residual/self.noise_active,residual/self.noise_active)
        penalty=jnp.vdot(values,prior)
        finite=jnp.all(jnp.isfinite(values))&jnp.all(jnp.isfinite(image))&jnp.all(jnp.isfinite(negative))&jnp.isfinite(chi2+penalty)
        return _CurrentPoint(values,linearization,image,residual,negative,chi2,penalty,.5*(chi2+penalty),finite)

    def _configuration(self):
        cfg=super()._configuration()
        cfg.update(n_data=self.projection.n_data,image_shape=self.projection.shape,
            fine_image_shape=self.renderer.image_shape,fine_n_data=self.renderer.n_active,
            observation_factor=self.projection.factor,observation_group_size=self.projection.group_size,
            effective_dpix=self.renderer.sim_config.dpix*self.projection.factor,
            PSF="original fine convolution followed by exact observation projection")
        return cfg

    def _result_products(self, joint_values, lambda_source, lambda_potential):
        source,potential=self._joint(joint_values)
        rendered=self.renderer.render(source,potential,macro_mapping=self.macro_mapping)
        active=self.projection.apply(rendered.image_active)
        residual=self.data_active-active
        chi2=jnp.vdot(residual/self.noise_active,residual/self.noise_active)
        penalty=jnp.vdot(joint_values,self.prior_matvec(joint_values,lambda_source,lambda_potential))
        return dict(source_values=source,source_x_axis=self.renderer.source_x_axis,
            source_y_axis=self.renderer.source_y_axis,source_bbox=self.renderer.source_bbox,
            potential_values=potential,potential_grid=self.renderer.potential_grid,
            deflection_x=rendered.deflection_x,deflection_y=rendered.deflection_y,
            convergence=rendered.convergence,derivative_valid=rendered.derivative_valid,
            fine_model_image=rendered.model_image,nonlinear_model_image=self.projection.full(active),
            nonlinear_image_active=active,nonlinear_normalized_residual=self.projection.full(residual/self.noise_active),
            data_active=self.data_active,noise_active=self.noise_active,chi2=chi2,
            regularization_penalty=penalty,objective=.5*(chi2+penalty),noise_logdet=self.noise_logdet,dtype=self.dtype.name)
