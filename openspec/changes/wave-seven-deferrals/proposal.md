# Proposal

## Why

Several Wave 3/4 catalogue entries mention `memmap2` backing or model persistence
(`SKKDTree`/`SKBallTree`, SVM kernel caches, tree persistence), but `memmap2` is not yet a
workspace dependency and the interop crate that will own persistence/export does not
exist. Without an explicit statement, readers of the PRD and of each change can mistake
these for omissions. The PRD should name them as Wave 7 (interop) deliverables and each
Wave 3/4 change should scope them out deliberately.

## What Changes

- Amend `docs/PRD.md` to state that `memmap2`-backed indexes and kernel caches,
  persistence and model export are delivered by the interop wave (Phase 7), not by the
  algorithm waves.
- Add a short cross-reference in the Phase 3 (neighbours/SVM) and Phase 4 (trees)
  sections so the deferral is visible where the catalogue mentions it.
- Update `CHANGELOG.md` under `Unreleased → Changed`.
- No code, no specs and no runtime behavior.

## Capabilities

### New Capabilities

None.

### Modified Capabilities

None. This is a documentation/scope clarification; `.openspec.yaml` sets
`skip_specs: true`.

## Impact

- **Docs:** `docs/PRD.md`, `CHANGELOG.md`, `docs/src/roadmap.md`, and the published
  `docs/documentations` roadmap cards.
- **Downstream:** the neighbours, SVM and tree changes cite the PRD section in their
  non-goals instead of silently dropping `memmap2`/export.
- **Acceptance criteria:** the amended PRD is internally consistent and renders in the
  book build.
