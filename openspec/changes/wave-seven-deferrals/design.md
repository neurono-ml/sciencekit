# Design

## Context

See `proposal.md` — Why. The wave plan names `memmap2` backing for trees, SVM kernel caches
and tree persistence, but `memmap2` is not a workspace dependency and `sciencekit_interop`
does not exist. The PRD's Phase 7 already owns safetensors/ONNX/JSON export and the I/O
traits; this change makes the dependency explicit instead of leaving it implied.

## Goals / Non-Goals

**Goals:**
- One PRD statement that names the deferred deliverables and their owning wave.
- Cross-references where the algorithm catalogue currently mentions them.
- A changelog entry so the scope clarification is visible to consumers.

**Non-Goals:**
- Adding `memmap2`, persistence or export code.
- Changing any spec or estimator behavior.
- Committing to a specific interop API in this change.

## Decisions

### Decision 1 — State the deferral in Phase 7 and reference it from Phase 3/4

Phase 7 is the owning wave for interop. The Phase 3 (neighbours/SVM) and Phase 4 (trees)
sections get a short "delivered by Phase 7" note where the catalogue mentions memory
mapping or persistence, so a reader does not infer an omission. The alternative — a
standalone deferrals document — would be less discoverable.

### Decision 2 — No spec delta

The change does not alter observable runtime behavior; it clarifies product scope. The
change sets `skip_specs: true`, following the schema's rule for documentation-only
changes. Inventing a requirement just to satisfy validation would be wrong.

### Decision 3 — Keep the wording implementation-neutral

The PRD note says *what* is deferred (memory-mapped backing, persistence, model export)
and *when* it arrives (Phase 7), not which crate/API will implement it. That keeps the
interop change free to choose the format and the index layout.

## Risks / Trade-offs

- **The deferral note becomes stale if the interop wave changes scope** → the interop
  change owns the PRD section and updates it when it lands.
- **Readers skip the Phase 7 note** → Phase 3/4 cross-references make it local.
- **Docs-only change drifts from the algorithm proposals** → each algorithm proposal
  cites the PRD section in its non-goals, and the independent review checks the wording.

## Migration Plan

Documentation-only. Reverting restores the previous PRD text and removes the changelog
entry.

## Open Questions

None. The exact interop format and index persistence layout belong to Phase 7.
