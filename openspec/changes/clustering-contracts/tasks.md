# Tasks

## 1. Contract

- [ ] 1.1 Write failing tests in `sciencekit_common` for `SKClustererPredictor` on a
  constant clusterer (dense and sparse input, assignment length, `-1` noise), then add the
  trait in `fit_traits/clusterer.rs` and re-export it; verify with
  `cargo test -p sciencekit_common clusterer`.
- [ ] 1.2 Update `SKUnsupervisedScorer` to the clusterer bound and `i64` assignments, and
  update the scorer-contract example tests; verify with
  `cargo test -p sciencekit_common scorer`.
- [ ] 1.3 Add the shared assignment-shape validation helper and tests for a length mismatch
  and accepted noise; verify with `cargo test -p sciencekit_common assignment`.

## 2. Documentation and review

- [ ] 2.1 Update `CHANGELOG.md` (English, keep-a-changelog 1.1.0) under `Unreleased →
  Changed` for the new assignment contract and the breaking scorer bound.
- [ ] 2.2 Add a short `Development Explained` subsection to the data-boundary or scoring
  chapter explaining why cluster assignments are `i64` and how noise is represented; verify
  with `mdbook build docs`.
- [ ] 2.3 Independent review by a separate agent validating every scenario in
  `clustering-contracts` and the modified `scoring-contracts` requirement; address every
  finding.

## Workflow follow-up

- Create the matching GitHub issue as an ADR in English and reference it from both PRs
  with a closing keyword.
- Push the code branch and the `-openspec` branch and open both PRs.
- Archive the change after both PRs merge, then delete the merged branches and worktrees.
