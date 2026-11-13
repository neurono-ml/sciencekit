# Design

## Context

See `proposal.md` — Why. `sciencekit` is a Rust API with generic builders and models. The
C ABI must erase generics (f32/f64, estimator families) behind handles and typed
descriptors. Python uses PyO3; this crate serves the C-callable languages.

## Goals / Non-Goals

**Goals:**
- A small, stable, versioned ABI.
- Memory safety across the boundary.
- One error/array convention for every language wrapper.

**Non-Goals:**
- High-level ergonomics, async, Python, GPU-specific APIs.
- Exposing every internal type.

## Decisions

### Decision 1 — Opaque handle table

Handles are `*mut SKFfiHandle` pointers to boxed enum variants:

```rust
enum SKFfiHandle {
    Builder(SKFfiBuilder),
    Model(SKFfiModel),
    Splitter(SKFfiSplitter),
    Workflow(SKFfiWorkflow),
}
```

`handle_create_*` returns an opaque pointer; `handle_free` consumes it. A null or
wrong-kind handle returns `SK_FFI_INVALID_HANDLE`. Freed handles are not dereferenced
(the ABI cannot fully detect use-after-free; bindings must follow the lifecycle rules and
the docs say so).

### Decision 2 — Status codes and thread-local error

`#[repr(C)] enum SKFfiStatus` maps `SKError` variants to stable codes
(`OK`, `INVALID_HANDLE`, `SHAPE`, `INVALID_ARGUMENT`, `UNSUPPORTED`, `NOT_FITTED`, `IO`,
`INTERNAL`). `sk_ffi_last_error()` returns a thread-local `*const c_char` valid until the
next call on that thread. Success clears it.

### Decision 3 — Array descriptors

```c
typedef struct {
  int32_t dtype;      // F32/F64/I64/U8
  const int64_t* shape;
  const int64_t* strides; // in elements, null = contiguous
  const void* data;
  int64_t len;
} SKFfiArray;
```

Inputs are borrowed for the call only. Outputs are owned arrays with a `sk_ffi_array_free`
function. The ABI validates dtype/shape/len/strides before constructing an ndarray view.

### Decision 4 — Generics erasure

Builders carry a runtime family/kind and a scalar dtype. `sk_ffi_builder_set_*` setters
update a configuration; `sk_ffi_fit` dispatches to the concrete family. Models store the
fitted model in an enum and expose predict/transform/score. Unsupported operations return
`UNSUPPORTED`.

### Decision 5 — Panic containment

Every `extern "C"` function wraps its body in `catch_unwind` and converts a panic to
`INTERNAL`; no Rust unwinding crosses the boundary. Unsafe blocks carry safety comments
and the crate denies unsafe-op-in-unsafe-fn.

### Decision 6 — ABI version and header

`sk_ffi_version()` returns `(major, minor)` and `sk_ffi_features()` returns a bitmask.
`cbindgen` generates `sciencekit.h` (or a hand-maintained header checked in CI). A C smoke
test compiles against the header and runs fit/predict.

### Decision 7 — File layout

```
crates/sciencekit_ffi/
  Cargo.toml (crate-type = ["cdylib", "staticlib"])
  cbindgen.toml
  include/sciencekit.h
  src/
    lib.rs, status.rs, error.rs, array.rs, handle.rs, builder.rs, model.rs,
    splitter.rs, workflow.rs, version.rs, *_tests.rs
  tests/c_smoke.c
```

Implementation files stay under 200 lines; the header is generated/checked.

## Risks / Trade-offs

- **Unsafe FFI** → small surface, safety comments, Miri/ASAN optional, C smoke test.
- **Use-after-free cannot be fully detected** → documented lifecycle and invalid-handle
  checks for null/wrong-kind.
- **String lifetimes** → thread-local error valid until next call; bindings copy it.
- **ABI stability** → version/feature query and additive-only changes within a major.
- **Generic erasure** → runtime kind/dtype dispatch with explicit unsupported errors.

## Migration Plan

Additive: new crate. No existing Rust API changes.

## Open Questions

None blocking. A Rust-generated C++ wrapper is a future convenience.
