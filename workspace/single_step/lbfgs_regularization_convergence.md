# Regularization and L-BFGS-B convergence in J0946 potential correction

## Run configuration

The comparison uses J0946 M3 sample 21342 and the Float64 dense L-BFGS-B fitter.
Each candidate starts from the saved M3 source and zero potential correction.
The macro mass model, image mask, noise, PSF, source grid, and adaptive source
weights are fixed. The image model uses cubic Hermite interpolation. Whitening
uses the saved M3 source with hard-box bilinear interpolation at zero potential,
with factors recomputed for each strength triple. The fit uses 2,293 selected
pixels, 10,000 source coefficients, and 1,134 potential coefficients at 0.1 arcsec
spacing. The per-candidate budget is 3,000 L-BFGS-B iterations.
[Run manifest](potential_lbfgs/manifest.json) · [Driver at `6fc2b20`](pt_iter_lbfgs.py)

The two DIRECT passes evaluated 277 distinct triples. The broad pass qualified
171 of 179. The refinement used a box extending 0.3 dex per strength around
the broad-pass winner and qualified all 98 new triples. The selected candidate,
216, used
\((\lambda_s,\lambda_0,\lambda_4)=(2.60815\times10^{-4},1.04545\times10^4,4.95231\times10^5)\).
It took 133 iterations and had local GN log evidence 6307.381623.
[Search summary](potential_lbfgs/completion.json) · [Selected fit](potential_lbfgs/maximum_evidence/result.json)

## Matched comparisons

Both comparisons hold \(\lambda_s=3.27415\times10^{-4}\) fixed. The first holds
\(\lambda_0=86.6608\) fixed and varies the fourth-difference strength. The second
holds \(\lambda_4=1971.55\) fixed and varies the amplitude strength.

| Candidate | Varied strength | Value | L-BFGS-B iterations | Outcome |
| ---: | --- | ---: | ---: | --- |
| 14 | \(\lambda_4\) | 1.97155 | 3,000 | Iteration limit |
| 6 | \(\lambda_4\) | 1,971.55 | 3,000 | Iteration limit |
| 78 | \(\lambda_4\) | 197,155.10 | 920 | Qualified |
| 10 | \(\lambda_4\) | 1,971,551.05 | 356 | Qualified |
| 77 | \(\lambda_4\) | 19,715,510.49 | 312 | Qualified |
| 6 | \(\lambda_0\) | 86.6608 | 3,000 | Iteration limit |
| 130 | \(\lambda_0\) | 1,867.05 | 1,089 | Qualified |
| 0 | \(\lambda_0\) | 8,666.08 | 502 | Qualified |
| 129 | \(\lambda_0\) | 40,224.39 | 237 | Qualified |
| 56 | \(\lambda_0\) | 186,705.09 | 131 | Qualified |
| 5 | \(\lambda_0\) | 866,608.27 | 65 | Qualified |
| 55 | \(\lambda_0\) | 4,022,439.25 | 26 | Qualified |

Within the broad pass, 8 of the 47 visited triples with
\(\lambda_4\leq 2{,}000\) hit the iteration limit. The median iteration count for
those 47 triples was 237, counting capped runs at 3,000. All 132 visited triples
above 2,000 qualified, with a median of 135.5 iterations. Six of the eight capped
fits also used \(\lambda_0=86.6608\). These counts describe the points visited by
this adaptive search.
[Candidate ledger](potential_lbfgs/evaluations.jsonl)

## Why weak regularization can slow the fit

With normalized residual \(r\), the objective for the source \(s\) and the
accumulated potential correction \(\delta\psi\) is

\[
F(s,\delta\psi)=\tfrac12\left(
\|r(s,\delta\psi)\|^2
+\lambda_s s^\mathsf{T}R_s s
+\lambda_0\|\delta\psi\|^2
+\lambda_4\|D_4\delta\psi\|^2
\right).
\]

- **\(\lambda_0\) controls amplitude.** Its penalty depends on the squared norm of
  the accumulated correction. Lowering it permits larger corrections at a given
  penalty cost. Smooth and oscillatory fields of equal norm receive the same
  amplitude penalty. The L-BFGS-B line search determines each update size.
- **\(\lambda_4\) controls fine-scale structure.** The fourth differences \(D_4\)
  penalize rapid node-to-node variations using complete five-node windows in
  grid-pixel units. Lowering this strength allows more of these variations at
  a given objective cost.

Some source and potential changes compensate each other in the predicted image.
Along a direction where \(J_s\Delta s+J_\psi\Delta\psi\) is small, the data term
supplies little Gauss–Newton curvature. Weak amplitude and smoothing penalties
can leave these directions shallow, so many updates may be needed to reach the
stopping criteria. Such weakly constrained structure can also absorb pixel
noise. The optimizer acts on the regularized posterior objective, and the
preferred reconstruction changes with the strengths.

The iteration counts above establish the trend for this run. The explanation in
terms of shallow directions follows from the objective structure. The fixed
reference whitening metric can also become less representative as the rays and
source change during optimization. Quantify the actual amplitude and roughness
of a candidate using \(\operatorname{RMS}(\delta\psi)\) and
\(\operatorname{RMS}(D_4\delta\psi)\) on its saved field and valid stencils.
[Objective and whitening implementation](../../TinyLensGpu/ObservationModel/LensImage/potential_correction_lbfgs.py)

## Interpreting the capped fits

The eight rows below reached 3,000 iterations with native L-BFGS-B status 1.

| Candidate | \(\lambda_s\) | \(\lambda_0\) | \(\lambda_4\) | GN stationarity statistic |
| ---: | ---: | ---: | ---: | ---: |
| 6 | \(3.27415\times10^{-4}\) | 86.6608 | 1,971.55 | \(1.034\times10^{-6}\) |
| 12 | \(1.51973\times10^{-5}\) | 8,666.08 | 1.97155 | \(5.347\times10^{-3}\) |
| 14 | \(3.27415\times10^{-4}\) | 86.6608 | 1.97155 | \(2.197\times10^{-1}\) |
| 73 | \(7.05394\times10^{-3}\) | 86.6608 | 1,971.55 | \(3.395\times10^{-7}\) |
| 74 | \(1.51973\times10^{-5}\) | 86.6608 | 1,971.55 | \(9.335\times10^{-2}\) |
| 99 | \(7.05394\times10^{-3}\) | 86.6608 | 1.97155 | \(9.233\times10^{-2}\) |
| 100 | \(1.51973\times10^{-5}\) | 86.6608 | 1.97155 | \(1.503\times10^{-1}\) |
| 154 | \(3.27415\times10^{-4}\) | 1,867.05 | 1.97155 | \(3.924\times10^{-9}\) |

At an endpoint, let \(g=\nabla F\) and \(H_{\rm GN}=J^\mathsf{T}J+R\), where
\(J\) is the normalized-residual Jacobian and \(R\) is the joint prior precision.
The best step in the local quadratic model is \(-H_{\rm GN}^{-1}g\). The
predicted decrease in \(F\) is the scalar
\(S_{\rm GN}=\tfrac12g^\mathsf{T}H_{\rm GN}^{-1}g\).
Candidates 6, 73, and 154 were below the \(10^{-3}\) threshold. Their
white-coordinate gradient infinity norms were \(4.73\times10^{-4}\),
\(1.76\times10^{-4}\), and \(8.21\times10^{-6}\), respectively. The native
gradient tolerance was \(10^{-8}\) and the relative objective-change tolerance
was \(10^{-15}\).
[Candidate ledger](potential_lbfgs/evaluations.jsonl)

The search used both successful native termination and the GN endpoint test
as qualification gates. This policy excluded the three capped candidates with
small \(S_{\rm GN}\) from evidence ranking. A finite endpoint with positive GN
curvature can still yield a diagnostic Laplace score. Its usefulness for
ranking depends on score stability, because the score depends on the endpoint
curvature determinant as well as the objective value.
[Qualification and evidence assembly](../../TinyLensGpu/ObservationModel/LensImage/potential_correction_lbfgs.py#L508)

## Reusing the comparison

Compare one strength at a time while holding the observation, starting fields,
reference interpolation, backend, and stopping controls fixed. Report capped
iterations alongside the endpoint GN statistic and the native exit reason.
For a capped candidate of scientific interest, rerun from the same start at the
same strengths with a larger iteration budget. Compare the endpoint objective,
local GN log evidence, and \(\delta\kappa\) over its derivative-valid domain.
Keep the field diagnostics alongside the timing or iteration counts so the
effect of regularization on the reconstructed field is visible.
