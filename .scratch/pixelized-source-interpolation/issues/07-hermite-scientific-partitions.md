# 07: Generate and reuse scientific partitions for Hermite sources

**What to build:** The maintained single-step and iterative scientific examples and their calibration flow can run with an explicit Hermite source and prepare, reuse, or explicitly specify a compatible block partition with recorded provenance.

**Blocked by:** 06 — Run single-step and iterative potential correction with Hermite.

**Status:** ready-for-agent
**State:** resolved

**Spec:** [Parent specification](../spec.md)

- [x] Both example flows carry their configured interpolation into reconstruction and exported metadata. Omitted settings and known historical bilinear calibrations retain the bilinear behavior.
- [x] Partition preparation supports sixteen source neighbors with four potential neighbors and validates the common ray count independently of the stencil widths.
- [x] Source tile votes sum the magnitudes of individual weights. A test with cancelling signed contributions verifies the intended incidence. Forward and transpose evaluations keep the signed interpolation weights.
- [x] Automatic partition provenance checks include geometry, interpolation, and vote rule. Verify reuse for matching cases and regeneration on mismatch, including the treatment of known old bilinear partitions.
- [x] A valid explicitly supplied fixed partition can be used for a controlled comparison and its origin is recorded. Exercise small example runs and their result records for both potential-correction methods.

## Comments

Claimed for implementation on `codex/psi-07-scientific-partitions`.

Implemented in `1b42c4b`, merged as `9041e4f`. The following command from
`examples/tests/` in `tinylens_gpu` on CPU with `MPLBACKEND=Agg` passed 23 tests.

```bash
pytest -xq test_potential_correction_calibration.py \
  test_potential_correction_scientific.py test_potential_correction_iterative_scientific.py
```

The final signed-vote/ray-count test passed after the added count assertion.
On CUDA, `pytest -xq test_potential_correction_calibration.py::test_bundled_legacy_partition_reuses_only_matching_bilinear_geometry`
passed and matched the bundled archive's geometry hash. The CPU geometry
differed and regenerated its partition. The same CUDA geometry with
Hermite rejected the archive's bilinear provenance. Tests also cover real
Hermite fits in all three drivers and intentional fixed cross-mode reuse.
