# Tasks

## 1. Kernel functions in `sciencekit_math`

- [x] 1.1 Add `kernels` module scaffold (`mod.rs` dispatcher, `config.rs` with `SKKernel`
  and `SKKernelGamma`) and tests for default gamma resolution `Scale`/`Auto`/`Value` and
  invalid parameters; verify with `cargo test -p sciencekit_math kernel`.
- [x] 1.2 Write and implement the linear kernel for dense inputs, with known dot products,
  shape, symmetry and a large-input test; verify with
  `cargo test -p sciencekit_math kernel`.
- [x] 1.3 Add sparse×sparse, sparse×dense and dense×sparse linear-kernel paths, with
  same-logical-matrix equivalence tests and a test proving no densification; verify with
  `cargo test -p sciencekit_math kernel`.
- [ ] 1.4 Write and implement the polynomial kernel (degree, gamma, coef0), including
  degree 1 equal to the linear kernel and a non-finite-input rejection test; verify with
  `cargo test -p sciencekit_math kernel`.
- [ ] 1.5 Write and implement the radial-basis-function kernel from the squared-distance
  dispatcher, including identical rows, distant rows, the negative-cancellation clamp and
  sparse/mixed equivalence; verify with `cargo test -p sciencekit_math kernel`.
- [ ] 1.6 Write and implement the sigmoid kernel with gamma and coef0, plus the
  same-logical-matrix equivalence across representations; verify with
  `cargo test -p sciencekit_math kernel`.
- [ ] 1.7 Add the precomputed-kernel validation wrapper (square matrix, matching row
  counts) and tests for a valid matrix, a non-square matrix and a row-count mismatch;
  verify with `cargo test -p sciencekit_math kernel`.
- [ ] 1.8 Add the row-parallel kernel path and a large-input parallel-equals-sequential
  test; verify with `cargo test -p sciencekit_math kernel`.

## 2. SVM crate scaffold and SMO core

- [ ] 2.1 Create `crates/sciencekit_svm` with `Cargo.toml`, `lib.rs` and dispatcher
  `mod.rs` files for `kernel`, `smo`, `svc`, `svr`, `probability`; add it to
  `[workspace.members]`; verify `cargo test -p sciencekit_svm` compiles an empty crate.
- [ ] 2.2 Implement the gradient state and second-order working-set selection in
  `smo/working_set.rs` and `smo/gradient.rs`, with unit tests on a hand-computed toy
  problem and a test that the selected pair maximizes the second-order gain; verify with
  `cargo test -p sciencekit_svm smo`.
- [ ] 2.3 Implement the KKT stopping criterion and the iteration/effort bookkeeping, with
  tests for a converged toy problem, an exhausted budget returning `NotConverged` and the
  reported iteration count; verify with `cargo test -p sciencekit_svm smo`.
- [ ] 2.4 Implement the bounded LRU kernel-row cache over dense and CSR training views,
  with tests for hit/miss, eviction, row recomputation and cache-size zero; verify with
  `cargo test -p sciencekit_svm cache`.
- [ ] 2.5 Implement the binary C-SVC SMO loop with clipping and bias recovery, and verify
  it separates a small separable data set and matches a hand-computed dual solution on a
  two-point problem; verify with `cargo test -p sciencekit_svm svc`.
- [ ] 2.6 Implement the epsilon-SVR SMO loop over the `2n` formulation, and verify a
  one-dimensional smooth response, the epsilon band and the support-vector count; verify
  with `cargo test -p sciencekit_svm svr`.

## 3. SVC estimator

- [ ] 3.1 Implement `SKSVCBuilder` with `C`, `kernel`, `degree`, `gamma`, `coef0`,
  `tolerance`, `maximum_number_of_iterations`, `cache_size`, `shrinking`, `class_weight`,
  `probability`, `random_seed`, `backend` and `execution_mode`; add default and
  invalid-value tests; verify with `cargo test -p sciencekit_svm svc_builder`.
- [ ] 3.2 Implement binary `SKSVC::fit` producing a model with support vectors, dual
  coefficients and bias, and verify the separable and overlapping fixtures; verify with
  `cargo test -p sciencekit_svm svc_fit`.
- [ ] 3.3 Implement one-vs-one multiclass fitting with pairwise classifiers and
  deterministic pair ordering; verify a three-class problem produces three binary models
  and each pair uses only its two classes; verify with `cargo test -p sciencekit_svm svc`.
- [ ] 3.4 Implement `predict_labels` by vote with smallest-index tie-breaking, and verify
  the three-class fixture and a hand-built vote tie; verify with
  `cargo test -p sciencekit_svm svc_predict`.
- [ ] 3.5 Implement `decision_function` with the documented OvO confidence layout and
  verify its shape, monotonic relation to labels and a binary fixture; verify with
  `cargo test -p sciencekit_svm svc_decision`.
- [ ] 3.6 Implement class weights, proving the balanced policy changes the fitted boundary
  on an imbalanced fixture and the explicit map is honored; verify with
  `cargo test -p sciencekit_svm svc_weights`.

## 4. Probability calibration

- [ ] 4.1 Implement Platt scaling with deterministic five-fold cross-validated decision
  values and regularized maximum likelihood; verify monotonicity, probability bounds and
  a hand-computed binary fixture; verify with `cargo test -p sciencekit_svm platt`.
- [ ] 4.2 Implement pairwise coupling from the binary probabilities, with convergence and
  iteration-cap tests on a three-class fixture; verify with
  `cargo test -p sciencekit_svm coupling`.
- [ ] 4.3 Wire `predict_probabilities` to Platt plus coupling, verify rows sum to one and
  values lie in `[0, 1]`, and verify `probability(false)` returns
  `UnsupportedOperation`; verify with `cargo test -p sciencekit_svm svc_probability`.

## 5. SVR estimator

- [ ] 5.1 Implement `SKSVRBuilder` with `C`, `epsilon`, kernel parameters, `tolerance`,
  `maximum_number_of_iterations`, `cache_size`, `shrinking`, `backend` and
  `execution_mode`; add default and invalid-value tests; verify with
  `cargo test -p sciencekit_svm svr_builder`.
- [ ] 5.2 Implement `SKSVR::fit` and `predict` with support-vector-only state, and verify
  a smooth one-dimensional response and the epsilon-band support-vector monotonicity;
  verify with `cargo test -p sciencekit_svm svr`.

## 6. Representation parity and error handling

- [ ] 6.1 Add dense, sparse and mixed fit tests for `SKSVC` and `SKSVR`, asserting
  same-logical-matrix predictions agree within tolerance and no sparse input is densified;
  verify with `cargo test -p sciencekit_svm representation`.
- [ ] 6.2 Add dense, sparse and mixed prediction tests against models fitted in each
  representation; verify with `cargo test -p sciencekit_svm representation`.
- [ ] 6.3 Add tests for empty input, single-class classification, feature-count mismatch,
  non-finite values and invalid kernel parameters, asserting the exact central error
  variant; verify with `cargo test -p sciencekit_svm validation`.

## 7. Determinism, concurrency and acceptance

- [ ] 7.1 Add tests proving repeated fits produce identical support vectors, dual
  coefficients and predictions, and that `random_seed` makes probability calibration
  repeatable; verify with `cargo test -p sciencekit_svm determinism`.
- [ ] 7.2 Add a concurrent prediction test sharing one fitted model across threads and a
  parallel-equals-sequential large-batch test; verify with
  `cargo test -p sciencekit_svm concurrency`.
- [ ] 7.3 Add `acceptance_tests/` covering tiny and large data (at least 10 000 rows),
  binary and multiclass classification, regression, every kernel, every representation
  and metric production through `sciencekit_metrics`; verify with
  `cargo test -p sciencekit_svm`.

## 8. Documentation, changelog and review

- [ ] 8.1 Update `CHANGELOG.md` (English, keep-a-changelog 1.1.0) under `Unreleased →
  Added` for the kernel module, `SKSVC`, `SKSVR`, the SMO solver and probability
  calibration; verify the entry lists every public artifact.
- [ ] 8.2 Companion documentation change on its own `docs/*` branch (merged into
  `docs/documentations`): add the kernel/SVM chapter to `docs/src/`, register it in
  `SUMMARY.md`, add a `Development Explained` chapter covering the kernel formulas, SMO,
  OvO, Platt/coupling and the support-vector model, update `maintaining.md`, and verify
  with `mdbook build docs`.
- [ ] 8.3 Record the Wave 7 deferral in `docs/PRD.md` — `memmap2`-backed kernel caches,
  persistence and model export are delivered by the interop wave, not by this change —
  and verify the amended paragraphs render in the book.
- [ ] 8.4 Run `cargo fmt --all --check`, `cargo clippy --workspace --all-targets -- -D
  warnings` and `cargo test --workspace`; fix every failure before pushing.
- [ ] 8.5 Independent review by a separate agent validating every scenario in
  `kernel-functions` and `support-vector-machines` against the code and tests; address
  every finding and re-run the gates.

## Workflow follow-up

- Create the matching GitHub issue as an ADR in English and reference it from both PRs
  with a closing keyword.
- Push the code branch and the `-openspec` branch and open both PRs.
- Archive the change after both PRs merge, then delete the merged branches and worktrees.
