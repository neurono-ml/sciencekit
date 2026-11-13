# Design

## Context

See `proposal.md` — Why. `sciencekit_common` owns `SKModelState`, `SKTensor` and metadata.
This change creates `sciencekit_interop` and uses `safetensors 0.8` (MSRV 1.80) for the
standard encoding. Sharding, padding and compression are extensions on top.

## Goals / Non-Goals

**Goals:**
- A safe, deterministic file format for every exported model.
- Standard Safetensors compatibility for the single-shard case.
- Sharding/padding/compression without changing the state contract.

**Non-Goals:**
- ONNX, JSON debug, Polars and Python.
- Cross-version tensor migration.
- Memory-mapped zero-copy reads (a future optimization).

## Decisions

### Decision 1 — `sciencekit_interop` crate

The crate owns format adapters. It depends on `sciencekit_common` for state and on the
format crates; it does not depend on model crates. This avoids a cycle with the retrofit
change.

### Decision 2 — Standard single-file encoding

The base file is a standard Safetensors file: 8-byte little-endian header length, a JSON
header with `dtype`, `shape`, `data_offsets` per tensor and `__metadata__` for our metadata.
The reader uses the `safetensors` crate for the standard path and our metadata parser for
the extended fields.

Alternative considered: a fully custom format (loses HuggingFace compatibility).

### Decision 3 — Sharding

A `SKShardPlan` splits tensors into shards by a byte threshold. Each shard is a valid
Safetensors file; an index JSON lists shard files and the tensor→shard map. The reader
reads every shard and merges by tensor name. The index is the only mutable file; shards are
immutable.

### Decision 4 — Header padding

A reserved `__padding__` region and a `__padding_bytes__` metadata field let a writer append
small tensors in place while space remains. When the addition would overflow, the writer
falls back to a new shard and updates the index. The padding bytes are excluded from tensor
data.

### Decision 5 — Compression

Compression wraps the complete file bytes (`gzip`, Brotli, Snappy) behind optional features
`compression-gzip`, `compression-brotli`, `compression-snappy`. The reader detects the
magic bytes and decompresses before parsing; a missing feature returns
`UnsupportedOperation`.

### Decision 6 — Validation and errors

Validate version, header length, offsets, tensor count, dtype/shape and shard index
completeness before constructing state. Errors: `InvalidHyperparameter`, `ShapeMismatch`,
`UnsupportedRepresentation`, I/O and compression errors.

### Decision 7 — File layout

```
crates/sciencekit_interop/src/
  lib.rs
  safetensors/{mod.rs, header.rs, writer.rs, reader.rs, shard.rs, padding.rs, compression.rs, *_tests.rs}
  error.rs
  acceptance_tests/mod.rs
```

`mod.rs` files are pure dispatchers; implementation files stay under 200 lines.

## Risks / Trade-offs

- **Custom extensions can break standard readers** → the single-shard path stays standard;
  padding/sharding are opt-in and documented.
- **Compression is whole-file, not per-tensor** → simple and correct; per-tensor
  compression is a future optimization.
- **Padding bookkeeping can corrupt offsets** → a dedicated padding module and corruption
  tests.
- **Shard index is mutable** → atomic write/rename; concurrent readers see a complete
  index.

## Migration Plan

Additive: new crate and workspace member. No existing API changes.

## Open Questions

None blocking. Memory-mapped reads and per-tensor compression are future changes.
