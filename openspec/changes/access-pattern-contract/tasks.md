# Tasks

Every task follows red-green-refactor: companion test first, confirmed failure, minimal
implementation, refactor with the gates green.

## 1. Vocabulary

- [x] 1.1 Add `SKAccessPattern::FullMemory` with its tier mapping and update every
  exhaustive `match`; verify with `cargo test -p sciencekit_common access_pattern`.
- [x] 1.2 Add the tier projection (`tier()` returning the parity tier) and tests for all
  four values; verify with `cargo test -p sciencekit_common access_pattern`.
- [x] 1.3 Add the declaration trait with `access_pattern` and `approximate_alternative`,
  plus tests asserting the default alternative is `None`; verify with
  `cargo test -p sciencekit_common declared_access_pattern`.

## 2. Declaration retrofit

- [ ] 2.1 Implement the declaration for the preprocessing and imputation estimators and
  remove their inline pattern assignments; verify with
  `cargo test -p sciencekit_preprocessing -p sciencekit_impute access_pattern`.
- [ ] 2.2 Implement the declaration for the linear-model estimators; verify with
  `cargo test -p sciencekit_linear_model access_pattern`.
- [ ] 2.3 Implement the declaration for decomposition, clustering, neighbors, trees,
  ensembles, SVM, outlier and pipeline estimators; verify with
  `cargo test --workspace access_pattern`.
- [ ] 2.4 Document the pipeline composition rule (weakest common pattern across stages) and
  test a mixed-pattern pipeline; verify with `cargo test -p sciencekit_pipeline access_pattern`.

## 3. Source compatibility

- [ ] 3.1 Add the shared source-kind validator with one failing test per incompatible pair
  and the accepted-pair table; verify with
  `cargo test -p sciencekit_common source_kind`.
- [ ] 3.2 Call the validator at the operation boundary before resolution in every estimator
  that accepts a source, and test that no element is read on rejection; verify with
  `cargo test -p sciencekit_common source_kind` plus the affected crate tests.
- [ ] 3.3 Implement the full-memory refusal (`UnsupportedOperation` naming the alternative)
  and test that a lazy source is rejected without consumption; verify with
  `cargo test --workspace full_memory`.

## 4. Observability

- [ ] 4.1 Add `access_pattern` to `SKOperationAttributes` and record it in the span; verify
  with `cargo test -p sciencekit_common operation_attributes`.
- [ ] 4.2 Update every operation call site to pass the declared pattern and test that fit
  and transform report the same value; verify with
  `cargo test --workspace access_pattern`.

## 5. Conformance

- [ ] 5.1 Add the registry macro and a workspace conformance test that fails for a missing
  declaration, a source-kind mismatch and a `FullMemory` entry without an alternative;
  verify each failure fixture.
- [ ] 5.2 Cross-check every declaration against its `parity.yaml` tier and fail with both
  values on drift; verify with `python3 scripts/check-parity-matrix.py` plus
  `cargo test --workspace access_pattern`.
- [ ] 5.3 Confirm the resolution behavior is unchanged (automatic never fails; explicit
  incompatibility keeps the same error) with the existing execution-planning tests; verify
  with `cargo test -p sciencekit_common execution`.

## 6. Documentation

- [ ] 6.1 Add a Development Explained chapter on the declaration replacing hardcoded
  values, the four tiers and the full-memory doctrine, grounded in the companion tests;
  verify `mdbook build docs` and registration in `SUMMARY.md`.

## Workflow follow-up

- Create the matching GitHub issue as an ADR in English and reference it from both PRs
  with a closing keyword.
- Push the docs branch and the `-openspec` branch and open both PRs.
- Archive the change after both PRs merge, then delete the merged branches and worktrees.
