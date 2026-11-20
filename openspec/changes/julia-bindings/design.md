# Design

## Context

See `proposal.md` — Why. `ffi-core` defines the C ABI; Julia can call it directly. The
package is thin: `ccall` declarations, wrappers and array conversion. No Rust-side Julia
dependency is needed (avoiding `jlrs` MSRV 1.97).

## Goals / Non-Goals

**Goals:**
- A small Julia package over the C ABI.
- Correct GC preservation and array conversion.
- Clear errors and library discovery.

**Non-Goals:**
- Julia-native estimators, GPU, async, zero-copy beyond array pointers.

## Decisions

### Decision 1 — `ccall` over `sciencekit_ffi`

`Project.toml` declares Julia 1.10+; `src/ScienceKit.jl` defines `const libsciencekit =
...` discovery (`ENV`, `Artifacts`, or a build step) and `ccall` signatures for status,
version, handles and arrays.

### Decision 2 — Wrappers and finalizers

Builders/models are Julia mutable structs holding a `Ptr{Cvoid}` handle. Constructors call
`*_create`; `finalizer` calls `*_free`. `close` is explicit and idempotent.

### Decision 3 — Arrays

`Array{Float32/Float64}` is passed by pointer with shape/strides; `GC.@preserve` wraps the
call. Non-contiguous views are copied or rejected. Outputs are copied into Julia arrays and
the native buffer is freed.

### Decision 4 — Errors

`SKFfiStatus` maps to a Julia exception subtype (`ScienceKitError`) carrying code/message.
The last-error message is copied before the next call.

### Decision 5 — File layout

```
julia/ScienceKit/
  Project.toml
  src/ScienceKit.jl
  src/{handles.jl, arrays.jl, errors.jl, estimators.jl}
  test/runtests.jl
```

## Risks / Trade-offs

- **GC moving arrays** → `GC.@preserve` and contiguous copies.
- **Library discovery** → documented env/artifact paths and a CI build step.
- **Finalizer ordering** → explicit `close` plus idempotent finalizer.
- **ABI changes** → version query at load.

## Migration Plan

Additive: new Julia package. No Rust API changes beyond the shared FFI.

## Open Questions

None blocking. A Julia-native wrapper for more families is future work.
