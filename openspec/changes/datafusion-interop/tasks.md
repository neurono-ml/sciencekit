## 1. Feature and streaming source

- [ ] 1.1 Add the `datafusion` feature and dependency; extend the CI feature matrix; document the supported DataFusion range.
- [ ] 1.2 TDD one-shot source: `execute_stream` batches convert through the Arrow adapter, final batch marked, row count and values preserved.
- [ ] 1.3 TDD reopenable source: a cloned DataFrame re-executes and delivers identical batches in identical order.
- [ ] 1.4 TDD projection pushdown: only selected columns are read; selection order preserved.
- [ ] 1.5 TDD filter pushdown: filtered rows never reach sciencekit; predicate applied before execution.

## 2. Async bridge behavior

- [ ] 2.1 TDD caller runtime responsiveness: a fit inside a multi-thread Tokio runtime leaves other tasks progressing.
- [ ] 2.2 TDD bounded buffering: at most depth + 1 batches resident between engine and consumer.
- [ ] 2.3 TDD deterministic order: four partitions produce the same concatenated rows as one partition.
- [ ] 2.4 TDD cancellation: dropping the source after each of the first three batches terminates the reader thread without panic.
- [ ] 2.5 TDD error propagation: a scan error after N batches returns a structured error and preserves the first N batches.

## 3. Registration and validation

- [ ] 3.1 TDD dense block registration as `MemTable`; query returns the same rows and names.
- [ ] 3.2 TDD CSR block registration in triple form; query returns the same logical values.
- [ ] 3.3 TDD supervised batch registration with a target column.
- [ ] 3.4 TDD streamed fit equals dense fit for a representative estimator; `f32` and `f64`.
- [ ] 3.5 TDD large and small datasets; generated Parquet larger than the memory budget.
- [ ] 3.6 Run the CI gates: `cargo fmt --all --check`, `cargo clippy --workspace --all-targets`, `cargo test --workspace`.
- [ ] 3.7 Add the DataFusion interop book chapter and the async-bridge `Development Explained` chapter; register in `SUMMARY.md`.
- [ ] 3.8 Update `CHANGELOG.md` and open the ADR issue for this change.
