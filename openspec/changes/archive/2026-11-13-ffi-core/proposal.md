# Proposal

## Why

Lua, Julia, Go and the JVM languages cannot call Rust generics directly. A stable C ABI is
the shared foundation: opaque handles, error codes, array descriptors and lifecycle
functions. Without it, every binding would re-invent unsafe glue and diverge on errors and
memory ownership. Python keeps its PyO3 path; the C ABI serves the other languages.

## What Changes

- New `sciencekit_ffi` crate producing a `cdylib`/`staticlib` with a versioned C ABI.
- Opaque handles for builders, fitted models, splitters and workflow objects; explicit
  create/free functions.
- Error model: integer status codes plus a last-error message and a per-thread error
  buffer.
- Array descriptors (`dtype`, `shape`, `strides`, `data`, `len`) for borrowed input and
  owned output buffers, with explicit release functions.
- Fit/predict/transform/score/model-load/save entry points for the supported families.
- Panic containment (`catch_unwind`) and no unwinding across the boundary.
- A generated C header (`cbindgen` or hand-written) and a C smoke test.
- ABI version/feature query so bindings can reject incompatible libraries.
- Explicit non-goals: high-level language ergonomics, Python (PyO3), async, GPU-specific
  handles.

## Capabilities

### New Capabilities

- `ffi-abi`: the C ABI surface, handles, error/array contracts, lifecycle, versioning,
  panic containment and thread-safety guarantees.

### Modified Capabilities

None.

## Impact

- **Code:** new `crates/sciencekit_ffi` workspace member.
- **APIs:** additive C ABI.
- **Dependencies:** no new core dependency; `cbindgen` optionally for headers.
- **Downstream:** unblocks Lua, Julia, Go and JVM bindings.
- **Acceptance criteria:** C smoke test, error/array round-trips, panic containment,
  handle lifecycle and concurrent calls.
