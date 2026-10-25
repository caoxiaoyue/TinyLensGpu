# Issue tracker: Local Markdown

Specs and tickets live in `.scratch/<feature-slug>/`. Read [the workspace index](../../.scratch/README.md) when locating existing work.

## Paths and publishing

- Spec: `.scratch/<feature-slug>/spec.md`.
- Implementation or incoming issue: `.scratch/<feature-slug>/issues/NN-<slug>.md`, numbered from `01`; one file per ticket.
- Decision map: `.scratch/<effort>/map.md`, with decision tickets under that effort's `issues/`.
- Append discussion under `## Comments`; link evidence and related records rather than duplicating them.

When a skill says to publish to the tracker, create the appropriate file above. Create directories only when they have content. When fetching a ticket, read its full body and comments, the parent spec or map, and any supersession notes. If a bare ticket number matches multiple features, resolve the ambiguity before working.

## Implementation tickets

Keep the upstream ticket headings and add a separate lifecycle field:

```markdown
# NN: Ticket title

**What to build:** The complete behavior this ticket delivers.

**Blocked by:** None (can start immediately), or sibling ticket numbers and titles.

**Status:** ready-for-agent
**State:** open

- [ ] Observable acceptance criterion
```

`Status` is the triage role from [triage-labels.md](triage-labels.md). Tickets generated from an approved spec are already `ready-for-agent`; incoming reports go through triage. Accept both `Status: value` and `**Status:** value` in existing records.

`State` records implementation progress:

| State | Meaning |
| --- | --- |
| `open` | Work remains and nobody has claimed it. |
| `claimed` | A session owns the work; record the branch or session in a comment. |
| `resolved` | The acceptance criteria have been met and relevant verification is recorded. |
| `closed` | Work ended without implementing this ticket, for example cancellation or supersession; record the reason and any replacement. |

Operations for `/implement` and `/implement-spec`:

1. Select a ticket with `Status: ready-for-agent` and `State: open` whose blockers are all `resolved`.
2. Set `State: claimed` and record ownership before starting. A claim in one worktree is not a cross-worktree lock; the coordinator must prevent duplicate assignments across checkouts.
3. On completion, record the verification command and outcome, evidence paths, and implementation commit when available under `## Comments`; then set `State: resolved`.
4. On cancellation or supersession, set `State: closed` and link the reason or replacement. A closed blocker does not automatically satisfy a dependency: update the dependent ticket's blocking edge to reflect the approved replacement or scope change.
5. Release unfinished claims by returning to `State: open` and recording remaining work. Update a parent spec's completion only after checking its full scope.

A legacy ticket without `State` is unclassified. Read its acceptance checklist, comments, parent supersession notes, and relevant implementation or verification records before assigning a lifecycle state. A triage label or checked checklist alone establishes neither readiness nor completion. Leave uncertain legacy work unclassified and report the gap.

## Wayfinding operations

Wayfinding tickets resolve decisions. They use `Type` and lifecycle `Status`, following the upstream local tracker convention, rather than implementation triage roles or `State`.

- **Map:** `.scratch/<effort>/map.md`, with the destination, notes, decisions so far, and remaining questions.
- **Child:** `.scratch/<effort>/issues/NN-<slug>.md`, with `Type: research`, `prototype`, `grilling`, or `task`; `Status: open`, `claimed`, or `resolved`; and `Blocked by: NN, NN` or `None`.
- **Frontier:** open, unclaimed tickets whose blockers are all resolved, in number order. Treat a legacy missing status as unclassified until its record is checked.
- **Claim:** set `Status: claimed` and record ownership before work; use the same cross-worktree coordination rule as implementation tickets.
- **Resolve:** append the answer under `## Answer`, set `Status: resolved`, then add a gist and link under the map's decisions so far.

## Research, experiments, and retained history

`.scratch/` is also a working area for temporary Python scripts, experiments, reproduction material, and results. These are repository conventions; the upstream local tracker template defines spec and ticket locations without prescribing a layout for experimental assets.

Each feature should have a short entry point, either a `README.md` or an existing note linked from `.scratch/README.md`, explaining the question, the useful files, and how to run them. Record required inputs, the applicable model and configuration, and the location of outputs. Distinguish historical experiments from checks of current behavior.

For new supporting material, optional `research/`, `experiments/`, and `output/` subdirectories can separate cited notes, scripts and small inputs, and generated artifacts. Create them only when useful. Keep existing files at their current paths unless a dedicated migration updates their callers, path assumptions, and documentation together.

Retain scripts and the small inputs or result summaries needed to understand or reproduce a recorded finding in version control. Logs, caches, and large generated outputs normally follow `.gitignore`. An ignored artifact is not automatically disposable: before cleanup, check references and preserve required evidence or document how to regenerate it, including inputs and configuration.

Reusable findings belong in `docs/technical-notes/`; validation reports with model, configuration, and provenance belong in `docs/validation/`. Link those reports to their supporting records. Move a check into `tests/` for core behavior or `examples/tests/` for example assembly when it should run as a maintained regression. Move a script into the relevant tool or example area when it becomes a maintained user workflow. A file's age alone is not a reason to move or delete it.

Archive in place: preserve ticket identifiers and paths, record lifecycle closure and any replacement, and retain a top-of-file supersession note for outdated spec sections. Read those notes before using historical requirements. Update `.scratch/README.md` when adding a feature or changing its main entry point; the source ticket or spec remains authoritative for its state and scope.
