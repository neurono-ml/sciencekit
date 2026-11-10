# Tasks

## 1. Contract

- [x] 1.1 Write failing tests in `sciencekit_common` for `SKClusterLabel` — cluster `0`,
  a non-zero cluster, noise, the four-byte size assertion and the `to_i64`/`from_i64`
  round-trip (including rejection below `-1`) — then add the type in
  `fit_traits/clusterer.rs`; verify with `cargo test -p sciencekit_common cluster_label`.
- [x] 1.2 Add `SKClustererPredictor<F>` returning `Array1<SKClusterLabel>` and a constant
  clusterer test covering dense and sparse input; verify with
  `cargo test -p sciencekit_common clusterer`.
- [x] 1.3 Update `SKUnsupervisedScorer` to the clusterer bound and typed assignments, and
  update the scorer-contract example tests; verify with
  `cargo test -p sciencekit_common scorer`.
- [x] 1.4 Add the shared assignment-shape validation helper and tests for a length mismatch
  and accepted noise; verify with `cargo test -p sciencekit_common assignment`.

## 2. Documentation and review

- [x] 2.1 Update `CHANGELOG.md` (English, keep-a-changelog 1.1.0) under `Unreleased →
  Changed` for the typed assignment contract and the breaking scorer bound.
- [x] 2.2 Add a short `Development Explained` subsection explaining the typed cluster
  label, the niche representation and the `-1` interop boundary; verify with `mdbook build
  docs`.
- [x] 2.3 Independent review by a separate agent validating every scenario in
  `clustering-contracts` and the modified `scoring-contracts` requirement; address every
  finding.

## Workflow follow-up

- Create the matching GitHub issue as an ADR in English and reference it from both PRs
  with a closing keyword.
- Push the code branch and the `-openspec` branch and open both PRs.
- Archive the change after both PRs merge, then delete the merged branches and worktrees.
