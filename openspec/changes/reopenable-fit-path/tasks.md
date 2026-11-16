# Tasks

Every task follows red-green-refactor: companion test first, confirmed failure, minimal
implementation, refactor with the gates green.

## 1. Contracts and harness

- [ ] 1.1 Add `sk_validate_reopenable_source` with a rejection test for a once-only
  provider; verify with `cargo test -p sciencekit_common reopenable`.
- [ ] 1.2 Add the multi-pass parity harness and a failing fixture for a provider that
  replays different data; verify with `cargo test -p sciencekit_common reopenable_parity`.

## 2. Clustering and decomposition

- [ ] 2.1 Implement `fit_streaming` for `SKKMeans` with center accumulation and parity
  tests; verify with `cargo test -p sciencekit_cluster kmeans_streaming`.
- [ ] 2.2 Implement `fit_streaming` for `SKPCA` over the three solvers; verify with
  `cargo test -p sciencekit_decomposition pca_streaming`.
- [ ] 2.3 Implement `fit_streaming` for `SKTruncatedSVD`; verify with
  `cargo test -p sciencekit_decomposition svd_streaming`.

## 3. Imputation

- [ ] 3.1 Implement `fit_streaming` for `SKKNNImputer` with the row-window guard; verify
  with `cargo test -p sciencekit_impute knn_streaming`.
- [ ] 3.2 Implement `fit_streaming` for `SKMissingIndicator`; verify with
  `cargo test -p sciencekit_impute indicator_streaming`.

## 4. Parity and observability

- [ ] 4.1 Run the parity harness per estimator on dense and CSR fixtures; verify with
  `cargo test --workspace reopenable_fit`.
- [ ] 4.2 Record the declared pattern and the provider kind in the operation span; verify
  with `cargo test -p sciencekit_common operation_attributes`.
- [ ] 4.3 Confirm in-memory behavior and tests are unchanged; verify with
  `cargo test --workspace`.

## 5. Documentation

- [ ] 5.1 Add a Development Explained chapter on multi-pass out-of-core fits, grounded in
  the parity harness tests; verify `mdbook build docs` and registration in `SUMMARY.md`.

## Workflow follow-up

- Create the matching GitHub issue as an ADR in English and reference it from both PRs
  with a closing keyword.
- Push the docs branch and the `-openspec` branch and open both PRs.
- Archive the change after both PRs merge, then delete the merged branches and worktrees.
