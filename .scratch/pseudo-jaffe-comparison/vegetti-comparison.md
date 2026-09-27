# Vegetti gravitational-imaging pseudo-Jaffe profile: primary-source notes

## Scope and bibliographic identity

This note uses only original papers, their arXiv records/PDFs, and publisher records.

- **Central paper:** S. Vegetti, L. V. E. Koopmans, A. Bolton, T. Treu & R. Gavazzi, *Detection of a dark substructure through gravitational imaging*, [arXiv:0910.0760](https://arxiv.org/abs/0910.0760), [PDF v2](https://arxiv.org/pdf/0910.0760), later MNRAS **408**, 1969–1981 (2010), [DOI](https://doi.org/10.1111/j.1365-2966.2010.16865.x), [publisher record](https://academic.oup.com/mnras/article/408/4/1969/1416697). The preprint was submitted/revised in October 2009; the journal article was published in 2010.
- **Vegetti & Koopmans 2009a:** *Bayesian strong gravitational-lens modelling on adaptive grids: objective detection of mass substructure in galaxies*, MNRAS **392**, 945–963, [arXiv:0805.0201](https://arxiv.org/abs/0805.0201), [PDF](https://arxiv.org/pdf/0805.0201), [DOI](https://doi.org/10.1111/j.1365-2966.2008.14005.x).
- **Vegetti & Koopmans 2009b:** *Statistics of mass substructure from strong gravitational lensing: quantifying the mass fraction and mass function*, MNRAS **400**, 1583–1592, [arXiv:0903.4752](https://arxiv.org/abs/0903.4752), [PDF](https://arxiv.org/pdf/0903.4752), [DOI](https://doi.org/10.1111/j.1365-2966.2009.15559.x).
- **Same-era cross-check:** S. Vegetti, O. Czoske & L. V. E. Koopmans, *Quantifying dwarf satellites through gravitational imaging: the case of SDSS J120602.09+514229.5*, [arXiv:1002.4708](https://arxiv.org/abs/1002.4708), [PDF](https://arxiv.org/pdf/1002.4708), [DOI](https://doi.org/10.1111/j.1365-2966.2010.16952.x).
- **dPIE comparison source:** A. Elíasdóttir et al., *Where is the matter in the Merging Cluster Abell 2218?*, [arXiv:0710.5636](https://arxiv.org/abs/0710.5636), [PDF](https://arxiv.org/pdf/0710.5636).

The reference list of arXiv:0910.0760v2 (printed p. 12) explicitly fixes the suffixes. **2009a is the adaptive-grid imaging-method paper; 2009b is the population-statistics paper.** The distinction matters:

- 2009a tests the method with **NFW** perturbers, not pseudo-Jaffe perturbers. Its NFW density is equation (46), printed p. 9 (journal p. 953), and the relevant description is “solely modelled with a NFW density profile.” It contains no pseudo-Jaffe equation.
- 2009b supplies the statistical inference for projected substructure fraction \(f\) and mass-function slope \(\alpha\); it is not the source of the imaging algorithm or pseudo-Jaffe equation.

Thus, 0910.0760's citations to 2009a concern the reconstruction/inference machinery and detection tests, not the analytic pseudo-Jaffe formula.

## Exact pseudo-Jaffe equations

### arXiv:0910.0760v2

Section 5.2, equation **(2)**, printed **p. 4** (final journal **p. 1974**), gives the circular projected convergence:

\[
k(r)=\frac{\alpha_{\rm sub}}{2}
\left[r^{-1}-\left(r^2+r_t^2\right)^{-1/2}\right].
\tag{2; arXiv v2}
\]

The TeX source and preprint print \(k(r)\), although the prose calls it the convergence, i.e. the dimensionless surface density normally denoted \(\kappa=\Sigma/\Sigma_c\). Immediately after equation (2), the paper gives

\[
r_t=\sqrt{\alpha_{\rm sub}\alpha},
\qquad
M_{\rm sub}=\pi r_t\alpha_{\rm sub}\Sigma_c.
\]

Definitions:

- \(r\): circular lens-plane radius measured from the subhalo centre;
- \(r_t\): substructure tidal radius;
- \(\alpha_{\rm sub}\): substructure lens strength;
- \(\alpha\): main-galaxy lens strength;
- \(\Sigma_c\): lensing critical surface density;
- \(M_{\rm sub}\): fitted pseudo-Jaffe mass parameter.

The final journal version changes notation but not the model: \(\alpha_{\rm sub}\to b_{\rm sub}\), \(\alpha\to b\), and \(k\to\kappa\):

\[
\kappa(r)=\frac{b_{\rm sub}}{2}
\left[r^{-1}-\left(r^2+r_t^2\right)^{-1/2}\right],\quad
r_t=\sqrt{b_{\rm sub}b},\quad
M_{\rm sub}=\pi r_tb_{\rm sub}\Sigma_c.
\]

This application does not sample \(r_t\) independently: the host strength and subhalo strength impose it through the square-root relation. The authors say this “leaves its total mass and position on the lens plane as free parameters.”

### Independent 2010 restatement

Vegetti, Czoske & Koopmans, arXiv:1002.4708, equation **(2)**, printed **p. 4**, writes the dimensional surface density explicitly:

\[
\Sigma(r)=\frac{\Sigma_c b_{\rm sub}}{2}
\left[r^{-1}-\left(r^2+r_t^2\right)^{-1/2}\right],
\qquad
r_t=\sqrt{b_{\rm sub}b},
\qquad
M_{\rm sub}=\pi r_tb_{\rm sub}\Sigma_c.
\]

It calls \(M_{\rm sub}\) the “mass within the tidal radius” and the only free G4 parameter. This confirms that the 0910.0760 normalization is a projected lensing-density normalization, not a 3D-density normalization.

## Total mass, aperture mass, and population mass

These are distinct quantities.

1. **Analytic pseudo-Jaffe lensing mass.** Direct integration of equation (2) over a circular projected aperture gives

   \[
   M_{2\rm D}(<R)=\pi b_{\rm sub}\Sigma_c
   \left[R-\sqrt{R^2+r_t^2}+r_t\right].
   \]

   This equation is a derivation from the printed profile, not an additional equation printed by Vegetti et al. Its limit is

   \[
   M_{2\rm D}(\infty)=\pi r_tb_{\rm sub}\Sigma_c=M_{\rm sub}.
   \]

   Therefore the fitted \(M_{\rm sub}\) is mathematically the profile's **finite total projected mass**. A literal projected aperture at \(R=r_t\) contains only

   \[
   M_{2\rm D}(<r_t)=(2-\sqrt2)M_{\rm sub}\simeq0.586M_{\rm sub}.
   \]

   The profile steepens smoothly; \(r_t\) is not a hard edge.

2. **Narrative shorthand.** The final 0910.0760 paper and arXiv:1002.4708 describe the fitted value as mass inside/within \(r_t\). That wording is not the literal circular-aperture integral above. Faithful use of their lens model should retain their defining normalization \(M_{\rm sub}=\pi r_tb_{\rm sub}\Sigma_c\).

3. **Finite-aperture measurements.** For SDSSJ0946+1006 the fitted total is \(M_{\rm sub}=(3.51\pm0.15)\times10^9\,M_\odot\) (68% CL), at \((-0.651\pm0.038,\,1.040\pm0.034)\) arcsec. Separately, the paper reports \((5.8\pm0.3)\times10^8\,M_\odot\) inside 0.3 kpc and \((1.1\pm0.05)\times10^9\,M_\odot\) inside 0.6 kpc (final journal p. 1977). These aperture masses are not synonyms for \(M_{\rm sub}\).

4. **Population projected fraction.** In 2009b, \(f(<R)\) is the cumulative dark-matter mass fraction in substructures inside a projected cylinder. Equations (4)–(5), printed p. 3, normalize expected counts through \(f(<R)M_{\rm DM}(<R)\) over \([M_{\min},M_{\max}]\). In 0910.0760 the reported \(f\simeq2.2\%\) applies only to \(4\times10^6\)–\(4\times10^9\,M_\odot\), not to all masses and not to a 3D virial fraction.

## Redshift and angular/physical conventions

For SDSSJ0946+1006, 0910.0760 adopts \(H_0=73\ {\rm km\,s^{-1}\,Mpc^{-1}}\), \(\Omega_m=0.25\), and \(\Omega_\Lambda=0.75\) (arXiv printed p. 2; journal p. 1970). The lens is at \(z_l=0.222\), the inner-ring source at \(z_{s1}=0.609\), and the outer source has \(z_{s2}<6.9\). The pseudo-Jaffe fit uses the inner ring; Ring 2 is excised, so the relevant critical density is for the \(z_l=0.222\), \(z_s=0.609\) geometry.

The modelling is angular:

- ring radii: \(1.43\pm0.01\) and \(2.07\pm0.02\) arcsec;
- standard image scale: 0.05 arcsec pixel\(^{-1}\) (a robustness data set uses 0.03 arcsec pixel\(^{-1}\));
- \(b\), \(b_{\rm sub}\), \(r\), \(r_t\), and subhalo coordinates use a common lens-plane angular unit; tables quote arcsec;
- interpretation converts radii to physical kpc at \(z_l\): the abstract reports \(r_{\rm tidal}=1.1\) kpc and the final journal abstract places the substructure 4.3 kpc from the galaxy centre.

The compact mass formula suppresses the angular-to-physical conversion. If \(b_{\rm sub}\) and \(r_t\) are angles in radians while \(\Sigma_c\) is physical mass per area, dimensional consistency requires

\[
M_{\rm sub}=\pi(D_Lr_t)(D_Lb_{\rm sub})\Sigma_c.
\]

Equivalently, one may use physical lens-plane radii or a critical density per angular area. This is a dimensional inference: 0910.0760 does not spell out the conversion beside equation (2), so arcsecond values must not be multiplied directly by a physical \(\Sigma_c\). Elíasdóttir et al. give the primary-source definition

\[
\Sigma_{\rm crit}=\frac{c^2}{4\pi G}\frac{D_S}{D_LD_{LS}}
\tag{A6; printed p. 15}
\]

with angular-diameter distances to the lens, to the source, and from lens to source.

The 2010 luminous-satellite application uses the same cosmology, \(z_{\rm FG}=0.422\), \(z_{\rm BG}=2.001\), and 0.05-arcsec pixels. Its two host models give \((2.78\pm0.04)\times10^{10}M_\odot\), \(r_t=0.68\) arcsec (PL+2SIS+PJ), and \((2.75\pm0.04)\times10^{10}M_\odot\), \(r_t=0.81\) arcsec (SIE+PJ), both on printed p. 4. The model-dependent \(r_t\) illustrates its tie to host strength.

## Conceptual comparison with a finite-core dPIE

Elíasdóttir et al. define a spherical dPIE with core radius \(a\), scale radius \(s>a\), and central density \(\rho_0\):

\[
\rho(r)=\frac{\rho_0}{(1+r^2/a^2)(1+r^2/s^2)}
\tag{A1; printed p. 15}
\]

and

\[
\Sigma(R)=\Sigma_0\frac{as}{s-a}
\left(\frac{1}{\sqrt{a^2+R^2}}-\frac{1}{\sqrt{s^2+R^2}}\right),
\qquad
\Sigma_0=\pi\rho_0\frac{as}{s+a}.
\tag{A3--A4; printed p. 15}
\]

Its total mass is

\[
M_{\rm TOT}=2\pi\Sigma_0as
\tag{A11; printed p. 16}.
\]

The Vegetti pseudo-Jaffe is the **zero-core/singular limit** of this spherical dPIE family: take \(a\to0\), identify \(s\leftrightarrow r_t\), and match \(\Sigma_0a\leftrightarrow\Sigma_cb_{\rm sub}/2\). Both are approximately isothermal between their inner and outer scales, steepen to \(\rho\propto r^{-4}\) (hence \(\Sigma\propto R^{-3}\)) outside the scale, and have finite total mass. The difference is central behavior: a finite-core dPIE has finite \(\rho(0)\) and \(\Sigma(0)\), whereas Vegetti's coreless profile has \(\Sigma\propto R^{-1}\) and the corresponding singular-isothermal \(\rho\propto r^{-2}\) cusp.

Elíasdóttir et al. explicitly caution that \(s\) is “not a 'cut-off' nor a trunctuation radius” (their spelling, footnote 9, printed p. 15). The same mathematical caution applies to Vegetti's smoothly truncated \(r_t\), despite the tidal-radius terminology.

## Bottom line

- The controlling formula is equation (2) of arXiv:0910.0760: a circular, singular, smoothly truncated projected profile.
- The paper uses \(r_t=\sqrt{b_{\rm sub}b}\); the current implementation deliberately adopts the Einstein-radius approximation described below instead.
- \(M_{\rm sub}=\pi r_tb_{\rm sub}\Sigma_c\) is the finite total projected lensing-mass normalization; keep it distinct from a finite-aperture mass and from any inferred pre-accretion virial mass.
- Do not attribute the pseudo-Jaffe equation to Vegetti & Koopmans 2009a: it supplies the method and uses NFW mock subhaloes. Do not confuse it with 2009b, which supplies population statistics.

## Adopted TinyLensGpu model choice

`PseudoJaffeSubhalo` implements a **Vegetti-style** singular Pseudo-Jaffe radial
profile, **not an exact reproduction** of Vegetti et al. (2010). Its intentional
tidal prescription is `rt = sqrt(b_sub * theta_E_host)`, using the host's sampled
Einstein-radius parameter directly. It does not convert that parameter to the
paper's main-lens density normalization, which is generally different for a
non-isothermal EPL. The profile shape is retained; the host-strength relation
is an adopted approximation with no independent truncation-radius parameter.

See the [recorded model choice](../../docs/technical-notes/untruncated-sis-subhalo-perturber-degeneracy.md#41-adopted-model-choice-einstein-radius-tidal-approximation).
The primary-source formulas above describe the paper, not a claim that the
pipeline duplicates all of its modeling assumptions or inference priors.
