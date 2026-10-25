# Subhalo prior audit

This directory retains research notes and diagnostic scripts for the J0946 subhalo comparison. It is supporting material for [the technical note and probe index](../../docs/technical-notes/untruncated-sis-subhalo-perturber-degeneracy.md); the scripts remain at their existing paths so recorded commands continue to locate them.

## Start here

- [Sensitivity study](sensitivity-study.md): the historical experiments, interpretations, and reproduction instructions. Its SIS numerical results and the current Pseudo-Jaffe pipeline describe different model configurations; follow the notes in its reproduction section.
- [Source-regularization literature](vegetti-source-regularization.md): cited research on the treatment of source regularization.
- [Technical note](../../docs/technical-notes/untruncated-sis-subhalo-perturber-degeneracy.md): the modeling context, limitations, and a table mapping each diagnostic script to its question.

## Running checks

Activate the repository environment as described in [AGENTS.md](../../AGENTS.md). From the repository root, the synthetic smoke check is:

```bash
PYTHONPATH=. python .scratch/subhalo-prior-audit/smoke_m4_m5.py
```

This script constructs synthetic inputs and checks the current M4/Pseudo-Jaffe M5 comparison without sampling or reading stage caches. It is a manually invoked check, outside the default pytest collection.

`probe_sis_farfield.py` is a standalone NumPy diagnostic of the untruncated SIS profile. Posterior-inspection and pipeline probes have their own cache requirements; follow their docstrings and the study's reproduction section for the working directory and output-path arguments. Historical SIS caches are not interchangeable with current Pseudo-Jaffe caches.

## Retention

Keep the notes and referenced scripts together. Run outputs and stage caches may live under the example pipeline's ignored output directories; a fresh checkout need not contain them. Before deleting or relocating supporting material, follow [the retention and promotion rules](../../docs/agents/issue-tracker.md#research-experiments-and-retained-history) and update its references. Retaining this material does not assert that every historical experiment has been rerun against the current code.
