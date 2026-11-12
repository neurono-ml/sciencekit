# Tasks

## 1. Silhouette

- [x] 1.1 Implement per-sample silhouette with chunked pairwise distances, noise exclusion
  and dense/sparse parity; verify with `cargo test -p sciencekit_metrics silhouette`.
- [x] 1.2 Implement the mean silhouette score and the fewer-than-two-clusters rejection;
  verify with `cargo test -p sciencekit_metrics silhouette`.
- [x] 1.3 Implement `SKSilhouetteScorer` over typed assignments and test scorer-equals-free-
  function; verify with `cargo test -p sciencekit_metrics silhouette_scorer`.

## 2. Pairwise labeling metrics

- [x] 2.1 Implement adjusted Rand index with identical/random/permuted-label fixtures;
  verify with `cargo test -p sciencekit_metrics adjusted_rand`.
- [x] 2.2 Implement normalized mutual information with arithmetic/geometric/max
  normalization and identical-label fixtures; verify with
  `cargo test -p sciencekit_metrics nmi`.
- [x] 2.3 Implement homogeneity, completeness and V-measure with hand-computed fixtures;
  verify with `cargo test -p sciencekit_metrics homogeneity`.

## 3. Centroid scores

- [x] 3.1 Implement Davies-Bouldin with compact/separated and noise-exclusion tests; verify
  with `cargo test -p sciencekit_metrics davies_bouldin`.
- [x] 3.2 Implement Calinski-Harabasz with compact/separated and noise-exclusion tests;
  verify with `cargo test -p sciencekit_metrics calinski_harabasz`.
- [x] 3.3 Add dense/sparse equivalence and no-densification tests for both scores; verify
  with `cargo test -p sciencekit_metrics clustering_representation`.

## 4. Validation, acceptance and review

- [x] 4.1 Add validation tests for empty input, length mismatch, single cluster, noise-only
  input and non-finite values; verify with
  `cargo test -p sciencekit_metrics clustering_validation`.
- [x] 4.2 Add large-data, concurrency and `Send + Sync` acceptance tests; verify with
  `cargo test -p sciencekit_metrics`.
- [x] 4.3 Run `cargo fmt --all --check`, `cargo clippy --workspace --all-targets -- -D
  warnings` and `cargo test --workspace`; fix every failure before pushing.
- [x] 4.4 Update `CHANGELOG.md` and the companion mdBook chapter (including a
  `Development Explained` chapter on silhouette, ARI/NMI and centroid scores); verify with
  `mdbook build docs`.
- [ ] 4.5 Independent review by a separate agent validating every scenario in
  `clustering-metrics`; address every finding and re-run the gates.

## Workflow follow-up

- Create the matching GitHub issue as an ADR in English and reference it from both PRs
  with a closing keyword.
- Push the code branch and the `-openspec` branch and open both PRs.
- Archive the change after both PRs merge, then delete the merged branches and worktrees.
