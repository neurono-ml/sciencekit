# Independent verification — low-rank adapters

Reviewer: separate agent, task 4.4. Inputs: code worktree `feat/lora-adapters`
(PR #399), openspec worktree `feat/lora-adapters-openspec` (PR #240), book
worktree `docs/lora-adapters-book` (PR #398). Date: 2026-11-13.

## Verdict

The implementation is numerically correct: the merged block equals
`W + (alpha / rank) · B A` to machine precision, apply and merge agree bit for
bit, and every probed invalid input returns a typed error. Two trust-boundary
defects were found and fixed test-first; all gates pass after the fixes.

## 1. Independent reference recomputation

A separate scratch program in `temporary/2026-11-13/lora-adapters/verify`
(Rust, path-depending on the worktree) fitted the base and adapter, dumped the
base `W`, factors `A`/`B`, metadata and query rows, and a Python/NumPy script
recomputed `W' = W + (alpha / rank) B A` and `X W'ᵀ + b` independently.

| case | seed | rank | alpha | heads × features | max \|W' − ref\| | max \|prediction − ref\| |
| --- | --- | --- | --- | --- | --- | --- |
| linear | 7 | 1 | 0.5 | 1 × 4 | 0.000e+00 | 1.110e-16 |
| linear | 19 | 3 | 3.0 | 1 × 3 | 0.000e+00 | 4.441e-16 |
| linear | 42 | 2 | 7.5 | 1 × 5 | 2.776e-17 | 4.441e-16 |
| logistic (binary) | 11 | 1 | 2.0 | 1 × 3 | 0.000e+00 | 8.882e-16 |
| logistic (3 classes) | 23 | 3 | 0.75 | 3 × 4 | 0.000e+00 | 4.441e-16 |

Maximum deviation over every case is `8.882e-16`, below the `1e-9` bound.
Rank 1, rank equal to the feature dimension and rank equal to the heads count
are all covered, and `alpha != rank` in every case.

## 2. Apply versus merge, sparse inputs, zero adapter

| check | result |
| --- | --- |
| \|apply − merge\| for the five cases above | exactly `0` (bit-identical vectors) |
| CSR-trained adapter: dense vs sparse vs merged dense vs merged sparse | all four vectors equal |
| zero adapter vs untouched base prediction | bit-identical for regression and for logistic scores and probabilities |
| merge → unmerge | original base coefficients, factors and predictions restored exactly |
| Safetensors file round trip (`adapter.safetensors`) | imported predictions bit-identical |
| JSON debug file round trip (`adapter.json`) | imported predictions bit-identical, equal to Safetensors |

Merge folds the same effective block prediction uses, so parity is exact rather
than tolerance-based.

## 3. Validation attacks

| attack | outcome |
| --- | --- |
| rank 0 in the builder | `InvalidHyperparameter` |
| negative rank (impossible through the `usize` builder) via persisted metadata | `InvalidHyperparameter` on import |
| `alpha = 0`, negative, `NaN` | `InvalidHyperparameter` |
| adapter target mismatch (logistic target on a linear base) | `UnsupportedOperation` |
| tree/ensemble family (`unsupported_family`) | `UnsupportedOperation` |
| unknown persisted target module | `InvalidHyperparameter` |
| transposed `A` shape (2×4 → 4×2) on import | `ShapeMismatch` |
| persisted rank disagreeing with stored tensors | `ShapeMismatch` |
| duplicate registry identifier | `InvalidHyperparameter`, no replacement |
| empty registry | valid, `number_of_adapters() == 0` |
| `NaN` / `+∞` / `−∞` in `A` or `B` on import | `NonFiniteValue` (fixed) |
| `training_loss` with wrong feature count | `ShapeMismatch`, previously an ndarray panic (fixed) |
| `training_loss` with short target/label vector | `ShapeMismatch`, previously a panic or silent truncation (fixed) |
| `training_loss` with non-finite features/targets | `NonFiniteValue` (fixed) |
| missing factor tensor | `InvalidHyperparameter` |

No probe panicked, and no partially applied adapter was observable: attachment
and import build a new value and fail before it is returned.

## 4. Convention audit

- `SK`/`sk_` prefixes are applied to every public struct, function and module;
  methods stay unprefixed; no abbreviations beyond the `SK`/`sk_` project prefix.
- Construction is builder-only (`SKLoraAdapterBuilder`); fields are private and
  there is no public constructor.
- `backend(SKBackendKind)` is a builder parameter (default `Faer`), carried into
  the attached and merged models, and every `fit`/`predict`/`decision_function`
  span records the configured kind instead of a hardcoded value.
- Dense and sparse inputs take the same public path, and every sparse branch uses
  `sk_csr_dense_product`/CSR traversal — no densification.
- `adapters/mod.rs`, `adapters/lora/mod.rs` and `fitting/mod.rs` are pure
  dispatchers; implementation lives in dedicated files.
- Every companion `*_tests.rs` is declared at the end of its `mod.rs`; no inline
  test blocks remain in implementation files.
- Iteration follows the repository style (`azip!` for element-wise accumulation,
  per-head matrix products as in the base logistic model, index-based argmax and
  sparse COO traversal where the index is the algorithm).

## 5. Findings fixed in this review

| finding | test | commit |
| --- | --- | --- |
| Import accepted `NaN`/infinite stored factors | `a_non_finite_stored_factor_is_rejected_on_import` | `d68e348` |
| `training_loss` panicked or silently truncated on shape mismatch and never checked finiteness | `training_loss_rejects_a_target_count_that_disagrees_with_the_rows`, `training_loss_rejects_a_feature_count_that_disagrees_with_the_base` | `09c4ff7` |

The book chapter `development-explained/ch66-lora-adapters.md` records both
boundaries (commit `3fbd486`).

## 6. Gates

- `cargo fmt --all --check`: clean.
- `cargo clippy --workspace --all-targets -- -D warnings`: clean.
- `cargo test --workspace`: exit 0, 45 test binaries green, no failures.
- `mdbook build docs`: succeeds (only the pre-existing large-search-index warning).
- `openspec validate lora-adapters --strict`: valid.
