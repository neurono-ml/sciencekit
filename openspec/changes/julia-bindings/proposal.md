# Proposal

## Why

Julia is the natural language for numerical computing. It can call the stable C ABI
directly through `ccall`, giving a small package without a second Rust binding layer. The
FFI core already defines handles, errors and array descriptors; Julia needs idiomatic
wrappers, GC preservation and package metadata.

## What Changes

- New Julia package `ScienceKit.jl` wrapping `sciencekit_ffi`.
- `ccall` declarations for builder/model/splitter entry points, status codes and array
  descriptors.
- Julia wrappers for builders, fitted models, fit/predict/transform/score, save/load and
  metrics.
- `Array{Float32/Float64}` conversion with `GC.@preserve`, shape checks and output arrays.
- Error mapping to Julia exceptions with status/message.
- Package layout (`Project.toml`, `src/`, `test/`), artifact/library discovery and CI.
- Explicit non-goals: Julia-native estimators, GPU, async, zero-copy beyond array pointers.

## Capabilities

### New Capabilities

- `julia-bindings`: package loading, ccall surface, array conversion, operations, errors
  and tests.

### Modified Capabilities

None.

## Impact

- **Code:** new Julia package (outside the Rust workspace) plus a thin Rust test harness if
  needed.
- **APIs:** additive Julia package.
- **Dependencies:** `sciencekit_ffi` shared library; Julia 1.10+.
- **Downstream:** exposes the library to Julia users.
- **Acceptance criteria:** package loads, every family fits/predicts on Julia arrays,
  errors map, GC preserve is correct and tests pass.
