# Real Euclid VIS qualification of adaptive-image Overlap

The maintained Euclid example ran on the VIS cutout for lens `102018665_NEG576731213511114485`. The 300×300 image and RMS and the 21×21 PSF matched the original multi-extension FITS arrays exactly. The example's additional-galaxy mask matched the corresponding file in the original cutout directory byte for byte. The FITS and example metadata agree on a pixel scale of 0.1 arcsec. The input file hashes are in the [retained results](../../.scratch/adaptive-image-overlap-evidence/experiments/euclid_vis_results.json).

## Run configuration

Stages A and L were rebuilt from the verified inputs with Nautilus seed 73. Stage A used 300 live points and a target of 2,000 effective posterior samples, and L used 200 and 800. They completed in 188.58 and 29.37 seconds on the RTX 5080. The upstream run used `346cc41113eda1409f2a460281da246b073b88f2`. The maintained A, B and L implementations were the same when the qualification fit ran at `1336264`. The later fit loaded the verified A and L products and regenerated a byte-identical 619-pixel Stage B feature mask. Stage L supplied the lens-light model subtracted before pixelized fitting.

M0 through M3 used Float32 on the RTX 5080 with JAX 0.7.1. They used first-order source precision, the 30×30 source grid, default bilinear interpolation, and four subpixels per side. The likelihood used explicit Overlap with tile width 10, PCG with preconditioner block width 10, a 400-iteration limit and relative tolerance \(10^{-5}\). The explicit inner batch width was five. The two marked images from A supplied the position penalty with separation threshold 0.1 arcsec.

M0 used two 200-point scans to choose the uniform regularization strength. M1 sampled EPL plus shear and the uniform source strength around the M0 value. M2 fixed the M1 mass and sampled \((\log\lambda,\rho)\). M3 used the M2 posterior medians for those strengths and varied the six EPL and shear parameters. Nautilus used an outer batch of 100 and seed 73 for all three sampled stages, with 300, 250 and 300 live points and a target of 600 effective posterior samples per stage.

## Stage results

The reporting point for M0 is the best grid-search value. The reporting points for M1 through M3 are the posterior medians. Each diagnostic uses the stage's saved configuration and a fresh solve at its reporting point.

| Stage | Likelihood calls | Posterior ESS | Saved score | Reporting-point `log_evidence` | Image `chi2` | PCG iterations | Wall time (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| M0 | 400 | — | 2560.5525 | 2560.5527 | 336.49 | 28 | 5.84 |
| M1 | 11,200 | 2,780 | 2567.9659 | 2584.8940 | 297.56 | 32 | 49.83 |
| M2 | 3,600 | 1,240 | 2594.9917 | 2599.6255 | 298.67 | 37 | 32.73 |
| M3 | 9,500 | 2,866 | 2587.8030 | 2601.3896 | 282.25 | 32 | 123.65 |

The saved M0 score is the best evaluated grid value. The saved M1 through M3 scores are Nautilus log evidence estimates. Every reporting-point solve and evidence evaluation was valid and the position penalty was zero. The M1 slope median was 1.8765, and M3's was 2.1572 with a 68% marginal interval of [2.0294, 2.2820]. M3's marked images mapped within 0.05662 arcsec in the source plane.

The M1 package contains a 300×300 PSF-convolved native source component and a distinct 30×30 source-plane raster. Re-rendering the native component from the saved source pixels at the M1 reporting point reproduced the saved reference within \(2.8\times10^{-9}\) in absolute intensity. M2 and M3 recorded the same reference identity, adaptation settings and fixed grid with bounds \((-0.8467671, 0.9027827, -0.8409336, 0.9086163)\) arcsec.

M2 selected posterior medians \(\log\lambda=-1.460518401169307\) and \(\rho=6.67540476303982\). The M3 artifact retained those medians exactly. The M3 objective recorded their Float32 values, \(-1.4605183601379395\) and \(6.6754045486450195\), and both diagnostic solves used \(\lambda=0.2321159244\). The applied source precision scales ranged from 2.301 to 792.668 in M2 and from 1.789 to 792.668 in M3 at the changed mass.

A second execution with the same settings restored the verified M1 reference and reused the matching M0, M2 and M3 objectives. The Stage A, L and M0 through M3 pickles retained their original SHA-256 hashes. The full fit and resume logs, stage pickles, native feature mask and diagnostic images are in the local ignored `.scratch/adaptive-image-overlap-evidence/output/vis` directory.

## Reproduction

Use a fresh checkout or worktree with the original multi-extension FITS and its `mask_extra_galaxies.fits` sidecar available. Link the four maintained example input FITS individually. Stage B writes its feature mask inside the example's `data` directory.

```bash
source ~/anaconda3/bin/activate
conda activate tinylens_gpu
export PYTHONPATH="$PWD"
export JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false
export MPLBACKEND=Agg OPENBLAS_NUM_THREADS=4

vis_source=/path/to/102018665_NEG576731213511114485.fits
vis_inputs=/path/to/maintained/euclid_examples/data
vis_example="$PWD/examples/pix_src_demo_operator/pipe/euclid_examples"
vis_output="$PWD/.scratch/adaptive-image-overlap-evidence/output/vis"
vis_runner="$PWD/.scratch/adaptive-image-overlap-evidence/experiments/euclid_vis_qualification.py"

mkdir -p "$vis_example/data" "$vis_output"
for name in image noise psf mask; do
    ln -s "$vis_inputs/$name.fits" "$vis_example/data/$name.fits"
done
python "$vis_runner" --phase upstream --source-fits "$vis_source" --out-dir "$vis_output" \
    > "$vis_output/upstream.log" 2>&1
cp "$vis_example/data/feature_mask.fits" "$vis_output/upstream_feature_mask.fits"

vis_command=(python -u "$vis_example/model.py" --skip-done --evidence-logdet overlap
    --evidence-block-size 10 --vectorized-chunk-size 5 --seed 73 --out-dir "$vis_output")
"${vis_command[@]}" > "$vis_output/vis_run.log" 2>&1
(cd "$vis_output" && sha256sum stage_{a,l,m0,m1,m2,m3}.pkl > before_resume.sha256)
"${vis_command[@]}" > "$vis_output/vis_resume.log" 2>&1
(cd "$vis_output" && sha256sum -c before_resume.sha256)
python "$vis_runner" --source-fits "$vis_source" --out-dir "$vis_output"
cp "$vis_example/data/feature_mask.fits" "$vis_output/feature_mask.fits"
```

The qualification runner verifies the native FITS against the original VIS HDUs and validates the saved stage products before writing the diagnostic record. [Runner](../../.scratch/adaptive-image-overlap-evidence/experiments/euclid_vis_qualification.py)
