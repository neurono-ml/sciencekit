## 1. Crate and neutral schema

- [x] 1.1 Create `crates/sciencekit_interop` and add it to the workspace members; add the crate to the CI feature matrix (core only).
- [x] 1.2 TDD `SKColumnRole` and `SKColumnSchema` with companion tests: role parsing, logical type naming, unknown logical type rejection.
- [x] 1.3 TDD `SKTabularSchema`: deterministic column lookup, duplicate-name detection, empty-schema rejection.
- [x] 1.4 TDD `SKColumnSelection`: feature order preservation, positional selection, unknown column, duplicate selection, empty feature set, more than one target.
- [x] 1.5 TDD `SKMissingValuePolicy`: default is reject; constant fill validates its value; categorical sentinel is the documented code.

## 2. Typed columns and batch

- [x] 2.1 TDD `SKColumnData` variants (`f32`, `f64`, `i64`, `bool`, nominal text, category codes) and their length/null accessors.
- [x] 2.2 TDD `SKTabularBatch`: uneven-length rejection, empty-batch handling, row count, column projection.
- [x] 2.3 TDD scalar coercion `f32`/`f64`/`i64`/`bool` to the model scalar, including the integer-magnitude guard for null promotion.
- [x] 2.4 TDD unsupported-column-type rejection once per batch, with the offending column named and a suggested conversion.

## 3. Conversion to library views and streaming blocks

- [x] 3.1 TDD dense materialization: multi-column table → owned row-major `Array2<F>`; verify values, order, and that source buffers are not aliased.
- [x] 3.2 TDD zero-copy path for a contiguous row-major input of the model scalar and for a single contiguous numeric column.
- [x] 3.3 TDD sparse materialization: CSR buffers → `SKFeatureBlock::Sparse`; verify zero preservation and byte size.
- [x] 3.4 TDD conversion to `SKDataBatch` and `SKSupervisedBatch`: ownership survives dropping the batch, position/final metadata preserved.
- [x] 3.5 TDD target extraction to `SKTargetView` (continuous, integer, nominal) and label canonicalization through the existing label table (round-trip and determinism).
- [x] 3.6 TDD categorical extraction: nominal and code columns through `SKCategoricalView`, including the missing sentinel.

## 4. Source contract, provider bridge, async bridge

- [x] 4.1 TDD `SKTabularSource` contract: reopenable and one-shot behavior, read-failure propagation as `SKError`.
- [x] 4.2 TDD adapter from `SKTabularSource` to `SKSourceProvider` / `SKSupervisedSourceProvider`: multiple passes over a reopenable source, one pass over a one-shot source.
- [x] 4.3 TDD async-to-sync bridge with a scripted async stream: ordered delivery, bounded resident batches (at most depth + 1), caller thread never polls.
- [x] 4.4 TDD bridge cancellation: dropping the consumer terminates the reader thread and drops the stream without panic or leak.
- [x] 4.5 TDD bridge error path: async stream error surfaces as the central taxonomy error after earlier batches were delivered.

## 5. Errors, observability, documentation

- [x] 5.1 Add the interop error variants to `SKError` (unknown/duplicate column, unsupported type, null under reject, uneven lengths, schema mismatch) with unit tests for messages.
- [x] 5.2 Add an interop conversion tracing span (source kind, rows, columns, policy) and test attribute recording.
- [x] 5.3 Run the CI gates: `cargo fmt --all --check`, `cargo clippy --workspace --all-targets`, `cargo test --workspace`.
- [ ] 5.4 Add a `Development Explained` chapter for the tabular seam and an interop chapter to the mdBook; register both in `SUMMARY.md`.
- [x] 5.5 Update `CHANGELOG.md` and open the ADR issue for this change.
