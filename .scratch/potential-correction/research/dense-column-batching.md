# Bounded dense response preparation

Ticket 11 identified an allocation risk in the existing all-column `vmap`
while assembling its 200×200/nsub=2 scientific case. With 3337 coefficients,
large simultaneous source-gather and subpixel response panels can exceed
the available 16 GiB device memory before the dense solve.

The dense reference now maps at most 16 basis responses simultaneously,
generates basis vectors within that map, and retains only the intended dense
design output. The interpolation, PSF, coupled system, prior and evidence are
unchanged. The helper is also available to the later iterative reference.
This is a bound on temporary column batching, not a measured whole-workflow
memory or speed claim. Ticket 11 owns its scientific-run verification.

Verification on 2026-10-29, in `tinylens_gpu`, from `tests/`:

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONPATH=/tmp/tinylens-pc-08 pytest -q \
  test_potential_correction_dense.py test_potential_correction_operator.py
```

Result: 33 passed in 29.59s. The public dense tests include independent
Gaussian evidence/field references, signed coupled solutions, explicit dtypes,
mask/subpixel/PSF behavior and failure cases. Operator comparisons preserve
the fixed-strength full-system agreement after the preparation change.
