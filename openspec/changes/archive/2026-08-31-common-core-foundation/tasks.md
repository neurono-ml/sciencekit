# Tasks — common-core-foundation

Every task follows TDD (the `tdd` skill): test first in a companion `*_tests.rs` module, confirmed failure, minimal implementation, refactoring. No file beyond 200 lines — the folder-module structure is already planned in the design.

## 1. Crate structure

- [x] 1.1 Create `crates/sciencekit_common` in the workspace with manifest, declared license and empty folder-module tree (`sk_float/`, `errors/`, `data_view/`, `target_view/`, `label_table/`, `fit_traits/`, `scorer_traits/`, `execution/`, `batching/`), compiling green
- [x] 1.2 Add base dependencies: `ndarray`, `sprs`, `num-traits`, `thiserror`, `sysinfo`

## 2. Scalar typing

- [x] 2.1 TDD for the sealed trait `SKFloat`: implemented by the supported standard floats, not implementable externally, integers rejected by bound

## 3. Error model

- [x] 3.1 TDD of the central enum with agreed variants (shape with expected/found, unsupported representation, identifiable invalid hyperparameter, mode incompatible with declared pattern, non-convergence with count, I/O with preserved source, conversion failure)
- [x] 3.2 TDD of automatic conversions from platform errors and verification of the derivation pattern for future algorithm enums

## 4. Data boundary

- [x] 4.1 TDD of the dense/sparse view: native copy-free conversions (borrowed dense, sparse CSR, owned block by borrowing), non-exhaustive marking
- [x] 4.2 TDD of the fallible seam: native types via direct conversions, a type with infallible conversion accepted via the std blanket, third-party fallible conversion propagating structured error into the operation's Result
- [x] 4.3 TDD of the target view: three variants (continuous, integer, nominal), lossless integer→continuous elevation, nominal referencing borrowed text
- [x] 4.4 TDD of precise rejection: a variant unsupported by a dense-only consumer produces a specific error before processing elements

## 5. Canonical labels

- [x] 5.1 TDD of deterministic canonicalization: nominal/integer sequences → compact indices + reversible table; full roundtrip; same input → same mapping
- [x] 5.2 TDD of the table as bearer of exportable data (readable metadata for the future model header)

## 6. Fit and transformation contracts

- [x] 6.1 TDD of the unsupervised trait: receives only features; return is the model associated type; fit on a shared reference keeps the estimator reusable
- [x] 6.2 TDD of the supervised trait: requires targets in the signature; an estimator implementing only the unsupervised trait rejects targets at compile time
- [x] 6.3 Compile-time tests proving predict-without-fit unrepresentability and `Send`/`Sync` of models produced by example contracts
- [x] 6.4 TDD of the transformer trait with typed output: compatible chaining validates statically; declared incompatibility detectable at compile time

## 7. Scoring contracts

- [x] 7.1 TDD of the supervised scorer: pure form over existing predictions without inference; provided form runs inference over an example contract and delegates, equivalent result
- [x] 7.2 TDD of the unsupervised scorer: pure form over existing assignments + analogous provided form
- [x] 7.3 TDD of fallibility: incomparable inputs produce structured taxonomy errors; genericity over the evaluated model allows reuse across families

## 8. Execution planning

- [x] 8.1 TDD of the intent enum (five modes) and consolidated plan struct
- [x] 8.2 TDD of pure resolution: same context → same plan; compatible explicit intent preserved in the plan
- [x] 8.3 TDD of hard error: explicit mode incompatible with declared pattern fails naming both sides, before processing data; automatic never produces this error
- [x] 8.4 TDD of per-operation resolution: distinct contexts between fit and prediction produce independent plans; simulated context dispenses physical reading (default constructor confined to real reading)

## 9. Streaming

- [x] 9.1 TDD of the owned block with metadata: survives source drop; exactly one final block on finite sources
- [x] 9.2 TDD of the sequential source: fallible iteration with structured error on intermediate failure
- [x] 9.3 TDD of the abstract random-access contract: direct positional access without scanning, without coupling storage mechanism

## 10. Acceptance and review

- [x] 10.1 Run all local gates (fmt, strict clippy, tests, doctests) and confirm green
- [x] 10.2 Verify this change's adapted acceptance checklist: contracts compile under `Send`/`Sync` where promised, complete companion coverage, mock data in ndarray/sprs, complete nomenclature with correct prefixes, no file beyond 200 lines
- [x] 10.3 Record on the PR that the full acceptance criteria from PRD §8.7/§10.3 are pending until the first estimator (Phase 1)
