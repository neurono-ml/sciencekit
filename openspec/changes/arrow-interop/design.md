# Design

## Context

See `proposal.md` — Why. `interop-foundation` defines `SKTabularSchema`, `SKColumnSelection`,
`SKMissingValuePolicy`, `SKColumnData`, `SKTabularBatch`, `SKTabularSource`, and the async-to-sync
bridge. This change supplies the Arrow implementation behind a feature flag.

Arrow's memory model: every array is a set of buffers (validity, data, offsets) plus a dtype,
logical length, and offset. A record batch is a schema plus one array per column. A table is a
schema plus one chunked array per column. A stream is an `ArrowArrayStream` of record batches.
Arrow stores **columns contiguously, not rows**, which is why a multi-column table cannot alias a
row-major matrix.

## Goals / Non-Goals

**Goals**

- Arrow as the single zero-copy interchange format for in-process adapters and language bindings.
- One reader path for all supported columnar formats.
- Correct null, dictionary, and string semantics.
- Ownership-safe C Data Interface in both directions.

**Non-Goals**

- No Arrow compute or query engine; selection and casts only.
- No nested (list/struct/map), temporal, decimal, or binary columns as numeric features in this
  change — they are rejected with a suggested conversion.
- No Arrow Flight, no database connectors, no Delta/Iceberg.
- No writer-side compression policy beyond what Parquet defaults provide.

## Decisions

### Depend on `arrow-*` crates, not on Polars/DataFusion

The crate depends on the Arrow crates directly so it can import Arrow data without pulling a query
engine. Polars and DataFusion adapters then convert their own types to Arrow and reuse this path.

### Zero-copy import policy

| Arrow column | Condition | Result |
|---|---|---|
| `Float32`/`Float64`/`Int64` primitive | no nulls, one buffer, no offset, dtype == scalar | borrow |
| primitive with nulls | any policy except reject | allocate + sentinel |
| dictionary | always | codes + label table (borrow values when possible) |
| utf8/large_utf8 | always | nominal view (borrow offsets/data) |
| boolean | role feature | bytes as 0/1 block |
| nested/temporal/binary | any | structured rejection |

Arrow column slicing (`Array::slice`) is zero-copy and shares buffers; the adapter propagates the
logical offset so sliced batches import correctly.

### C Data Interface as the binding ABI

Export uses `FFI_ArrowArray::new(&ArrayData)` and `FFI_ArrowSchema::try_from(&DataType)`; import
uses `unsafe { from_ffi(array, &schema) }` with the caller guaranteeing conformance. Streams use
`FFI_ArrowArrayStream` with a producer-owned `private_data` and a release callback that drops the
boxed Rust state. The exported struct is `Send + Sync` per arrow-rs; the release callback is the
single owner-transfer point. Tests must exercise release exactly once, including error paths.

Alternative considered: a bespoke C ABI of `(data, indices, indptr, shape)`. Rejected because it
would have to be reimplemented in every binding and would not carry null bitmaps or schemas.

### File readers through Arrow's format crates

- Parquet: `ParquetRecordBatchReaderBuilder` with `with_batch_size`, `with_projection`, row-group
  selection; a path-based factory produces a reopenable source.
- CSV: `arrow_csv::reader::Format` (infer schema from a prefix) then `ReaderBuilder`.
- JSON/NDJSON: `arrow_json::ReaderBuilder` with an explicit schema (inference is opt-in).
- IPC/Feather: `arrow_ipc::reader::FileReader` / `StreamReader`.
- Each source owns a `File`, returns `SKTabularBatch`, and maps `ArrowError` into `SKError`.

Projection pushdown is a real performance requirement, not an optimization: a 500-column Parquet
file must not decode unselected columns.

### Export direction

`Array2<F>` → `PrimitiveArray<F>`; `CsMat<F>` → three arrays (indptr, indices, data) plus shape
metadata; `SKDataBatch` → `RecordBatch`; `SKSupervisedBatch` → `RecordBatch` with a target column.
`SKDataView` borrows, so export from a view copies into an Arrow buffer; export from an owned block
moves the buffer into `Buffer::from_vec` where possible.

## Risks / Trade-offs

- **Arrow crate version coupling** (Polars/DataFusion pin Arrow versions) → Mitigation: adapters
  depend on the same Arrow version as the engine where possible; otherwise convert through the C
  Data Interface, which is version-independent.
- **`unsafe` FFI surface** → Mitigation: isolate in `ffi.rs`, wrap in safe constructors, run Miri
  where feasible, and test release-once with a counting allocator.
- **Parquet projection bugs** → Mitigation: tests assert that unselected columns are never decoded
  and that selection order is preserved.
- **Null-vs-NaN ambiguity** → Mitigation: policy is explicit; the reject default keeps behavior
  predictable.
- **Feature compile time** → Mitigation: Arrow behind one feature; core and default builds do not
  compile it.

## Migration Plan

Additive. Feature `arrow` defaults off. Later adapters enable it. No existing code changes.

## Open Questions

- Whether to expose Arrow's `arrow-csv` schema inference or require an explicit schema by default.
  Deferred: the design defaults to inference with an explicit-schema override.
