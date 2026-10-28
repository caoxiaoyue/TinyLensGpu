# 08: Dense iterative reconstruction at fixed strengths

**What to build:** A caller can jointly update the pixelized source and potential with a fixed-strength dense Gauss–Newton/LM reconstruction, receiving the final nonlinear image, explicit termination status, iteration history, and local-linearization evidence.

**Blocked by:** 07 — Operator tuning and single-step backend calibration.

**Status:** ready-for-agent
**State:** claimed

**Parent:** [Approved potential-correction spec](../spec.md).

- [ ] A separate iterative entry point starts from truth-derived source-node intensities and zero potential correction with fixed source/potential geometry and macro parameters.
- [ ] Current pixelized-source spatial gradients analytically differentiate the existing bilinear brightness formula and retain its cell/boundary/out-of-bounds conventions.
- [ ] Each required linearization updates corrected ray positions, source mapping, and current source gradients; the initial truth gradient is not reused as the ongoing gradient.
- [ ] Source and potential increments solve the coupled damped Gauss–Newton/LM problem, including the negative prior-gradient term for the accumulated fields.
- [ ] Acceptance is based on the actual nonlinear data misfit plus the accumulated source/potential prior, and accepted updates reduce that objective within justified numerical tolerances.
- [ ] Damping, objective-change, scaled-update, gradient, and iteration-limit controls are explicit and recorded.
- [ ] Convergence, iteration-limit termination, repeated rejection/stalling, failed linear solves, and nonfinite states produce distinguishable outcomes.
- [ ] Successful final evidence uses complete dense determinants of the local Gauss–Newton system without LM damping and is labeled as a local-linearization approximation.
- [ ] The common result includes the final nonlinear image, residuals, valid-domain maps, and useful accepted/rejected iteration history.
- [ ] Python manages control flow while JAX performs expensive numerical work; large systems are not solved with NumPy.
- [ ] Tests cover within-cell source gradients, objective acceptance, cumulative-prior behavior, cache refresh, damping-independent final evidence, and failure paths.
- [ ] A small real fixed-strength reconstruction demonstrates iterative recovery independently of outer optimization or full scientific plotting.

## Scope and handoff

The dependency on ticket 07 is the explicitly approved stage gate. This ticket
does not repeatedly apply and sum the single-step solution. Operator iterations
are ticket 09 and nested hyperparameter search is ticket 10.

## Comments

- 2026-10-29: Claimed by the ticket-08 implementer on branch `codex/pc-08`,
  worktree `/tmp/tinylens-pc-08`, from integration `35d0318`. Ticket 07 is
  resolved with its calibration report and retained summary. The approved
  spec already confirms the public result/operator testing seams; development
  uses focused red/green numerical slices. Operator iteration stays in 09.

