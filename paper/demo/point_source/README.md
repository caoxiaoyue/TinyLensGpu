# Point Source Demo

End-to-end demo for `PointSourceProbModel` using lensed image positions of a point source.

## Run

From this folder, first generate mock data then run inference:

```bash
python sim_data.py
python run_model.py
```

Optional fast mode (no Nautilus):

```bash
POINT_SOURCE_DEMO_N_EFF=300 python run_model.py
```

If `run_model.py` reports missing data, regenerate with:

```bash
python sim_data.py
```

## What it does

1. `sim_data.py` builds a ground-truth lens model (`SIE + Shear`)
2. `sim_data.py` solves synthetic image positions and adds Gaussian noise
3. `run_model.py` loads `data/point_source_positions.npz`
4. `run_model.py` fits lens + source parameters with `PointSourceProbModel`
5. Uses Nautilus nested sampling

## Outputs

`sim_data.py` writes to `data/`:

- `point_source_positions.npz`: packed input data for inference
- `observed_positions.csv`: observed image positions
- `true_positions.csv`: noiseless image positions
- `position_sigma.csv`: per-image uncertainties
- `point_source_data.png`: quick-look data plot

`run_model.py` writes to `output/`:

- `result_samples.csv`: posterior samples
- `result_summary.csv`: q16/q50/q84 summary
- `results.pkl.gz`: serialized result bundle
- `point_source_positions.png`: observed vs predicted position plot
