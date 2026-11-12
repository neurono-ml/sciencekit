# Proposal

## Why

The PRD's internal model format is an extended Safetensors file: a JSON header with
metadata, tensor payloads, optional sharding, header padding for small appends and
compressed variants. `interop-contracts` and `model-export-retrofit` now give every model a
state; this change turns that state into a portable file and reads it back.

## What Changes

- New `sciencekit_interop` crate with a Safetensors writer/reader over `SKModelState`.
- Standard-compatible single-file Safetensors encoding (8-byte header length, JSON header,
  tensor payloads).
- Extended metadata: estimator name, format version, scalar dtype, class tables and typed
  hyperparameters in the header.
- Sharding: split large states into immutable `.safetensors` shards plus a JSON index;
  the reader merges shards.
- Padding: reserve header/payload space so small tensor additions can be written in place
  while padding lasts; otherwise a new shard is created.
- Compression wrappers: `.safetensors.gz`, `.safetensors.brotli` and `.snappy.safetensors`
  behind optional features.
- Import: read a file (or shard set) back into `SKModelState`; native model import belongs
  to `model-export-retrofit`.
- Validation: header/tensor consistency, offsets, dtype/shape, duplicate shards, truncated
  files and compression errors.
- Explicit non-goals: ONNX, JSON debug, Polars/DataFusion and Python.
- Companion tests: round-trip, sharding, padding, compression, large/small models,
  concurrency and corrupt-file errors.

## Capabilities

### New Capabilities

- `safetensors-format`: the extended Safetensors encoding, metadata, sharding, padding,
  compression and round-trip validation.

### Modified Capabilities

None.

## Impact

- **Code:** new `crates/sciencekit_interop` workspace member; `sciencekit_common` state
  consumed, no changes to model crates.
- **APIs:** additive.
- **Dependencies:** `safetensors 0.8` (MSRV 1.80), `serde`, `serde_json`, optional
  `flate2`, `brotli`, `snap`.
- **Downstream:** unblocks JSON debug (shared crate), Polars sources (shared crate) and the
  Python bindings' model persistence.
- **Acceptance criteria:** round-trip fidelity, large and small models, concurrent reads,
  corruption errors and documented format/version compatibility.
