# Tasks

## 1. PRD amendment

- [x] 1.1 Add the Phase 7 scope statement to `docs/PRD.md`: `memmap2`-backed indexes and
  kernel caches, persistence and model export are delivered by the interop wave, not by
  the algorithm waves.
- [x] 1.2 Add the cross-reference note to the Phase 3 (neighbours/SVM) and Phase 4
  (trees/ensembles) sections where the catalogue mentions memory mapping or persistence.
- [x] 1.3 Verify the amended paragraphs are consistent with the wave-plan catalogue and
  with the non-goals of the neighbours and SVM proposals.

## 2. Changelog and verification

- [x] 2.1 Update `CHANGELOG.md` (English, keep-a-changelog 1.1.0) under `Unreleased →
  Changed` for the clarified Phase 7 scope.
- [x] 2.2 Run `mdbook build docs` and verify the PRD-derived documentation renders
  without errors.

## 3. Review

- [x] 3.1 Independent review by a separate agent validating that the deferral is stated
  once, referenced where relevant, and does not contradict any spec or in-flight proposal.

## Workflow follow-up

- Create the matching GitHub issue as an ADR in English and reference it from both PRs
  with a closing keyword.
- Push the docs branch and the `-openspec` branch and open both PRs.
- Archive the change after both PRs merge, then delete the merged branches and worktrees.
