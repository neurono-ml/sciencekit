## 1. Feature and in-memory import

- [ ] 1.1 Add the `arrow` feature and dependencies (`arrow-array`, `arrow-schema`, `arrow-buffer`, `arrow-cast`, `arrow-select`, `arrow-ipc`, `arrow-csv`, `arrow-json`, `parquet`); extend the CI feature matrix.
- [ ] 1.2 TDD schema mapping: Arrow `DataType`/`Field` → `SKColumnSchema`; reject nested/temporal/binary with a structured error naming the field.
- [ ] 1.3 TDD `RecordBatch` import: names, order, row count, values; empty batch; sliced batch with non-zero offset.
- [ ] 1.4 TDD `Table`/`ChunkedArray` import: chunks concatenated in order, total length, zero chunks treated as empty column.
- [ ] 1.5 TDD primitive zero-copy: `f32`/`f64`/`i64` columns without nulls borrow the Arrow buffer; assert pointer identity.
- [ ] 1.6 TDD primitive materialization: nulls produce NaN/sentinel/fill per policy; integer magnitude guard rejects lossy promotion.
- [ ] 1.7 TDD boolean import as 0/1 feature and as categorical codes.
- [ ] 1.8 TDD dictionary and utf8 import: codes + label table, nominal view borrows text, deterministic label order.

## 2. Streaming import and sources

- [ ] 2.1 TDD `RecordBatchReader` → `SKTabularSource`: batch size respected, final batch marked, error propagation.
- [ ] 2.2 TDD reopenable file source factory: two passes produce identical batches in identical order.
- [ ] 2.3 TDD projection: unselected columns are not decoded; selection order is preserved.
- [ ] 2.4 TDD source → `SKSourceProvider` / `SKSupervisedSourceProvider` through the foundation adapter; multi-epoch and one-shot behavior.

## 3. C Data Interface

- [ ] 3.1 TDD export a dense `Array2`/`SKDataBatch` as `FFI_ArrowArray` + `FFI_ArrowSchema`; consume it with a second import and assert equality.
- [ ] 3.2 TDD export an `FFI_ArrowArrayStream` over a source; import it back and assert batch-for-batch equality.
- [ ] 3.3 TDD ownership: release callback called exactly once (counter); imported array survives producer release; error path releases once.
- [ ] 3.4 TDD dictionary and utf8 round-trip through the C Data Interface.
- [ ] 3.5 Run the FFI tests under Miri or a sanitizer job where available.

## 4. File formats

- [ ] 4.1 TDD Parquet source: row-group and batch-size bounds, projection pushdown, truncated file error, empty file.
- [ ] 4.2 TDD CSV source: inferred schema, explicit schema override, delimiter/header options, malformed row error.
- [ ] 4.3 TDD Arrow IPC/Feather source: file and stream readers, truncated stream error after complete batches.
- [ ] 4.4 TDD JSON/NDJSON source: explicit schema, inference opt-in, malformed document error.
- [ ] 4.5 TDD large-file behavior with a generated file larger than the configured memory budget; assert bounded resident batches.

## 5. Export and validation

- [ ] 5.1 TDD export dense predictions and transformed blocks as Arrow arrays/record batches.
- [ ] 5.2 TDD export CSR blocks: indptr/indices/data and shape metadata preserved.
- [ ] 5.3 TDD `f32`/`f64` cast paths and reporting of `zero_copy = false`.
- [ ] 5.4 Run the CI gates: `cargo fmt --all --check`, `cargo clippy --workspace --all-targets`, `cargo test --workspace`.
- [ ] 5.5 Add the Arrow interop book chapter and the C Data Interface `Development Explained` chapter; register in `SUMMARY.md`.
- [ ] 5.6 Update `CHANGELOG.md` and open the ADR issue for this change.
