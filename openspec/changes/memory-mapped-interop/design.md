# Design

## Context

See `proposal.md` — Why. The existing `SKMappableSource<F>` trait (spec `streaming-batches`) requires
`number_of_rows()` and `row(index) -> ArrayView1<'_, F>` with O(1) positional access. The
`SKLazySource<F>` trait requires owned batches. `interop-foundation` provides the conversion and
error plumbing; `arrow-interop` provides Arrow batch conversion.

`memmap2::Mmap` gives a `&[u8]` over the file and is `Send + Sync`. A row view can be built with
`ndarray::ArrayView1::from_shape` over a slice created from the mapped bytes when alignment and
element size permit. Arrow IPC files store record batches with an offset table at the end of the
file, which enables indexing without decoding all data.

## Goals / Non-Goals

**Goals**

- True O(1) row access for fixed-width matrices.
- Zero-copy views into the mapping for the matching scalar type.
- One mapped file usable both sequentially and by random access.
- Clear failure when a file does not match its declared layout.

**Non-Goals**

- No memory-mapped CSV or JSON (not fixed-width; those stream through `arrow-interop`).
- No memory-mapped Parquet random row access (row groups are the unit; streaming is the contract).
- No `.npz` archives, no compressed `.npy`, no Fortran order.
- No big-endian raw files; `.npy` endianness is honored only for little-endian.

## Decisions

### Raw binary needs an explicit descriptor

A raw file has no metadata. The source takes rows, columns, and dtype from a sidecar JSON descriptor
(`{"rows": ..., "columns": ..., "dtype": "f64"}`) or from explicit builder parameters. The byte
length must equal `rows * columns * size_of::<F>()`. This avoids guessing and makes corruption
detectable.

Alternative: infer shape from file length — rejected because any factor pair would be plausible.

### `.npy` header parsing with `npyz`

`npyz` parses the magic, version, header length, dtype descriptor, shape, and order. The adapter
validates `fortran_order == false` and a supported dtype, then maps the payload. The dtype must
match the model scalar for a zero-copy view; otherwise the caller must opt into a cast (which
materializes per batch).

### Arrow IPC batch index

`arrow_ipc::reader::FileReader` exposes the footer's record-batch blocks. On open, the adapter reads
the footer only, records `(first_row, row_count, block_offset)` per batch, and defers batch decode
until access. A row lookup binary-searches the cumulative offsets, decodes that batch, and returns
the row. Sequential streaming decodes batches in order with the configured block size. This is O(1)
in the number of batches for the lookup plus the cost of decoding one batch.

### Borrowed rows and lifetimes

`row(index)` returns `ArrayView1<'_, F>` tied to the source. Because `Mmap` outlives the view, this
is sound. The source must not be moved in a way that invalidates the mapping; `Mmap` is stable when
boxed or stored behind `Arc`. The adapter stores `Arc<Mmap>` and derives views from it.

### Alignment

`ndarray` requires properly aligned data. `memmap2` mappings are page-aligned at the start; raw
payloads that begin at offset zero are aligned, but `.npy` payloads follow a header whose length is
not necessarily a multiple of the element size. The adapter checks alignment and, when misaligned,
returns a structured error suggesting padding or copying. Tests cover both aligned and misaligned
headers.

## Risks / Trade-offs

- **File mutated while mapped** → Mitigation: document that mutation is undefined; validation at
  open; tests use immutable fixtures.
- **SIGBUS on truncated file** → Mitigation: validate length before exposing views; map with
  `unsafe` only after the length check; tests read a file truncated after open and document the
  limitation.
- **`.npy` alignment** → Mitigation: alignment check and explicit error; a copy fallback can be
  added later behind a flag.
- **IPC index memory for millions of batches** → Mitigation: store three `usize` per batch;
  document the bound; batch counts are far smaller than row counts in practice.
- **Platform differences (Windows file locking)** → Mitigation: tests target Linux/macOS in CI;
  document Windows read-only mapping.

## Migration Plan

Additive and feature-gated. No existing code changes.

## Open Questions

- Whether to support `.npz` (ZIP of `.npy`) later. Deferred; not part of this change.
