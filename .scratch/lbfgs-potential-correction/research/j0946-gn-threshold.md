# J0946 L-BFGS-B endpoint threshold comparison

Two Float32 operator runs optimized the three regularization strengths with DIRECT using the same J0946 M3 handoff, cubic Hermite source, 0.1 arcsec potential grid and principal-block evidence. The requested global and refinement budgets were 160 and 90 evaluations. The manifests match apart from `gn_stationarity_tol`. Both runs used an RTX 5080, JAX 0.7.1 and SciPy 1.17.1. Each pass stopped at its evaluation budget and the workflow retained the highest qualified score it visited.

| Quantity | GN tolerance 0.01 | GN tolerance 10000 |
| --- | ---: | ---: |
| Qualified / evaluated | 256 / 279 | 254 / 267 |
| GN threshold rejections | 10 | 0 |
| Selected source strength | 0.0003565623 | 0.0003387764 |
| Selected fourth-difference strength | 82609.55 | 82609.55 |
| Selected amplitude strength | 2411.39 | 2411.39 |
| Selected GN stationarity | 0.006983 | 0.002991 |
| Selected local GN log evidence | 6403.145 | 6403.088 |
| Peak δκ at (−0.7, 1.1) arcsec | 0.3340 | 0.3281 |
| Noise-normalized image residual RMS | 0.6247 | 0.6228 |

The relaxed run qualified six candidates with GN above 0.01. The largest GN among its qualified candidates was 0.04445. The best score among the six was 6372.121. Eight other candidates reached the L-BFGS-B iteration limit and five failed the recomputed PCG residual check.

The library and J0946 search entry points default to `10000` so GN remains a diagnostic and its finite cutoff is effectively inactive for typical fits. [Current fit policy](../../../docs/guides/potential-correction.md#fixed-strength-l-bfgs-b)

The maps have Pearson correlation 0.99979 and RMS difference 0.00215 over all 802 derivative-valid nodes. The RMS residual within 0.3 arcsec of the published perturber coordinates, over 112 fitted pixels, was 0.6154 and 0.6139, respectively, compared with 0.9995 at the saved M3 start. The global RMS uses the same 2,293 pixels in each run. This comparison measures recovery of the upper-arc convergence feature conditional on the fixed macro model. [Vegetti et al. (2010)](https://pure.rug.nl/ws/files/72160205/mnras0408_1969_1_.pdf)

Float64 operator refits at the two selected strength triples kept the same peak node, with heights 0.3502 and 0.3285 and log evidence 6402.424 and 6403.026, respectively. Among 232 common triples qualified by both Float32 runs, the median absolute difference in log evidence was 0.0195. The ranking of these two selected triples switched under Float64 refitting.

## Reproduction

Run from the repository root with the saved `workspace/slacs0946_potential_correction/macro` handoff. The run manifests and result records identify the handoff and reconstructions by SHA-256.

```bash
source ~/anaconda3/bin/activate && conda activate tinylens_gpu
export JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false
export OPENBLAS_NUM_THREADS=1 NVIDIA_TF32_OVERRIDE=0 MPLBACKEND=Agg
study=.scratch/lbfgs-potential-correction/output/j0946-gn-threshold-repeat
python -u -m workspace.slacs0946_potential_correction.pt_iter_lbfgs_operator_float32 \
    --output "$study/gn-1e-2" --gn-stationarity-tol 0.01
python -u -m workspace.slacs0946_potential_correction.pt_iter_lbfgs_operator_float32 \
    --output "$study/gn-1e4" --gn-stationarity-tol 10000
```

[Retained comparison](../output/j0946-gn-threshold-ab/comparison.json) · [Convergence maps](../output/j0946-gn-threshold-ab/convergence_comparison.pdf) · [Float64 refit log](../output/j0946-gn-threshold-ab/refit_selected.log)
