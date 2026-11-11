# Design

## Context

See `proposal.md` — Why. `ffi-core` defines the C ABI. Go can call it through cgo; the
wrapper layer handles handles, arrays, errors and finalizers. The Go module lives outside
the Cargo workspace and is tested with the native library built from the FFI crate.

## Goals / Non-Goals

**Goals:**
- A small Go API over the C ABI.
- Safe array conversion and lifecycle.
- Module packaging and tests.

**Non-Goals:**
- Go-native estimators, async, GPU, zero-copy beyond cgo pointers.

## Decisions

### Decision 1 — cgo wrapper

`sciencekit.go` declares the C types/functions via a `/* #cgo ... */` preamble and a
`types.go` with Go wrappers. `sciencekit_ffi` is built as a shared library and discovered
via `CGO_LDFLAGS`/`LD_LIBRARY_PATH` or a documented build step.

### Decision 2 — Arrays

A matrix is a flat `[]float64`/`[]float32` plus rows/cols; the wrapper builds a contiguous
copy if needed and passes a pointer with a `C.malloc` copy. Outputs are copied into a new
Go slice and the native buffer is freed.

### Decision 3 — Errors and lifecycle

`SKFfiStatus` maps to a `ScienceKitError` with `Code`/`Message`. Builders/models hold a
`typed `*C.SKFfiBuilder` / `*C.SKFfiModel`` handle; `Close` frees it; `runtime.SetFinalizer` frees if the user forgets. Calls
on a closed handle return an error.

### Decision 4 — File layout

```
go/
  go.mod
  sciencekit.go
  types.go
  arrays.go
  errors.go
  *_test.go
  examples/
```

## Risks / Trade-offs

- **cgo build/link complexity** → documented flags and CI.
- **Finalizer timing** → explicit Close preferred; idempotent finalizer.
- **Pointer passing rules** → use `unsafe.Pointer` only for the call duration with
  a `C.malloc` copy.
- **ABI changes** → version query at init.

## Migration Plan

Additive: new Go module. No Rust API changes beyond the shared FFI.

## Open Questions

None blocking.
