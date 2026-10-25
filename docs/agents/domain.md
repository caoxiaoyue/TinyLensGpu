# Domain Docs

This repository uses one domain context for strong-lens modeling and inference.

## Read for the task

- Before exploring domain behavior, read [`GLOSSARY.md`](../../GLOSSARY.md) and the ADRs in [`docs/adr/`](../adr/) that concern the task.
- When locating code, imports, or tests, read [`codebase-map.md`](codebase-map.md).
- When finding a task's requirements or history, follow [`issue-tracker.md`](issue-tracker.md).

If a domain file does not yet exist, continue with available sources. Add terms and ADRs when a discussion actually resolves them.

## Document roles

- `GLOSSARY.md` defines domain terms and preferred vocabulary.
- `docs/adr/` records decisions and their trade-offs. Surface conflicts with a relevant ADR before changing the decision.
- `docs/guides/` describes supported workflows and operating instructions.
- `docs/technical-notes/` records reusable engineering and modeling findings.
- `docs/validation/` records validation results for a stated model and configuration.
- `.scratch/` holds feature requirements, work records, research, and reproduction material under the tracker conventions.

Use glossary terms in task descriptions and code discussions. An absent term is a question to resolve when needed, not a reason to invent a synonym.

For a spec-based review, use the explicitly designated spec or ticket and read its supersession notes. Research notes and validation reports provide evidence; they become requirements only when explicitly designated as such. Preserve the model, configuration, and provenance attached to numerical findings.
