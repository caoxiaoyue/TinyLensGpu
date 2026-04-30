# INFERENCE KB

## OVERVIEW
`TinyLensGpu/Inference/` is the sampler-facing wiring layer. It turns `ParamU`-driven models into priors, likelihood callables, nested samplers, and optimizers.

## STRUCTURE
```text
TinyLensGpu/Inference/
|- param_u.py             # ParamU parameter wrapper
|- build_prior.py         # unit-cube transforms + PriorSpec
|- build_likelihood.py    # sampler-facing callable builders
|- base.py                # shared inference base class
|- Optimizer/             # DIRECT / DE / basin hopping wrappers
`- NestedSampler/         # Nautilus / Dynesty / UltraNest wrappers
```

## WHERE TO LOOK
| Task | Location | Notes |
|------|----------|-------|
| Parameter metadata / modes | `TinyLensGpu/Inference/param_u.py` | dynamic, static, linear |
| Prior extraction and transforms | `TinyLensGpu/Inference/build_prior.py` | walks Caskade modules |
| Likelihood wrapping / vectorization | `TinyLensGpu/Inference/build_likelihood.py` | `jax.vmap` lives here |
| Global optimization wrapper | `TinyLensGpu/Inference/Optimizer/` | SciPy-backed search |
| Nested sampling wrapper | `TinyLensGpu/Inference/NestedSampler/` | Demo-facing sampler adapters |

## CONVENTIONS
- Modern inference is programmatic only: build model objects in Python, then call `make_prior_transformation()` and `make_likelihood()`.
- `ParamU` is the canonical inferred parameter type; preserve support for `prior_type`, `prior_settings`, and hard `limits`.
- `make_likelihood(..., vectorized=True)` is the expected high-throughput path for nested samplers.
- Sampler and optimizer adapters should stay thin wrappers around the probability-model API.

## ANTI-PATTERNS
- Do not assume YAML config pipelines still exist in this tree; current demos are direct Python.
- Do not introduce new parameter wrappers that bypass `ParamU` semantics.
- Do not break compatibility with `ImageProbModel` callable expectations when changing wrappers.

## NOTES
- `NestedSampler/__init__.py` is intentionally empty; import concrete wrappers directly when editing that area.
- Demos under `paper/demo/` are the best real-world examples of this package in use.
