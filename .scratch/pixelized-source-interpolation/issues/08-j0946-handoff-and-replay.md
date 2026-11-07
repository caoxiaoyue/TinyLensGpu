# 08: Carry source interpolation through J0946 handoffs and replay an archived state

**What to build:** The active J0946 research handoff and its single-step and iterative consumers agree on the saved source interpolation. A fixed-state replay measures agreement between the experimental and production Hermite implementations on saved J0946 fields and rays.

**Blocked by:** 03 — Save and resume a staged fit with its source interpolation, and 06 — Run single-step and iterative potential correction with Hermite.

**Status:** ready-for-agent
**State:** resolved

**Spec:** [Parent specification](../spec.md)

- [x] Active handoff production and consumption use the shared interpolation provenance contract. A known older library handoff with no mode field constructs a bilinear source, and a recorded Hermite choice is carried into the configured renderer.
- [x] Preserve the recorded hard-box, experimental zero-ring, and Hermite boundary conventions when reading archived inputs. Retain the original source grid and fields.
- [x] Replay a documented saved cubic reconstruction with its fixed source intensities, potential coefficients, corrected rays, precision, and imaging settings. Evaluate the experimental and library Hermite paths with the same inputs.
- [x] Compare ray brightness, spatial derivatives, and the prediction after subpixel integration and PSF convolution. Record the archive identifier, input provenance, command, tolerance, and observed differences as a fixed-state comparison.

## Comments

Claimed for implementation on `codex/psi-08-handoff-replay`.

Implemented in `196f5fa`, merged as `fed3762`. The
[J0946 replay report](../../../docs/validation/pixelized-source-interpolation-j0946-replay.md)
records the commands, checksum-pinned inputs and numerical comparisons
for the Float64 cubic trial and the two archived boundary controls.
The Hermite ray brightness and PSF-convolved image matched to
`6.66e-16` and `2.22e-16` absolute, respectively. The maximum
spatial derivative error was `5.51e-14`, within the `1e-11` tolerance.

From `examples/tests/`, the CUDA `pytest -xq test_j0946_single_step_workflow.py`
run passed 43 tests. From `workspace/iteration/tests/`, the CPU
`pytest -xq test_j0946_iteration.py test_j0946_cubic_source.py test_j0946_zero_boundary.py`
run passed 13. A final CPU selection on the merged ticket tree passed 15
handoff, export, resume and archive cases.

After the support-edge precision fix in `ee1c67d`, the refreshed cubic replay
matched the experimental derivatives and PSF-convolved image within
`1.60e-12` and `1.50e-15` absolute, respectively. The hard-box and zero-ring
controls also passed. The report and raw records were refreshed in `38e37fd`.
