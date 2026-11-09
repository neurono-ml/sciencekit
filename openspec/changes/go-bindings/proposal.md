# Proposal

## Why

Go is common for services and data tooling. cgo can call the stable C ABI directly, and a
thin Go module with typed wrappers avoids requiring users to write unsafe cgo themselves.
The FFI core supplies handles, errors and array descriptors; Go adds idiomatic types,
finalizers and module packaging.

## What Changes

- New Go module `github.com/neurono-ml/sciencekit-go` (outside the Cargo workspace) over
  `sciencekit_ffi`.
- cgo bindings for status/version/handles/arrays and fit/predict/transform/score/save/load.
- Go wrapper types for builders/models with `Finalizer` cleanup and explicit `Close`.
- `[]float64`/`[]float32` matrix conversion with shape validation and output copying.
- Error mapping to Go errors carrying status/message.
- Module packaging (`go.mod`), examples, tests and CI with the native library.
- Explicit non-goals: Go-native estimators, async, GPU, zero-copy beyond cgo pointers.

## Capabilities

### New Capabilities

- `go-bindings`: cgo surface, typed wrappers, array conversion, errors, lifecycle and
  packaging.

### Modified Capabilities

None.

## Impact

- **Code:** new Go module and cgo files; no Rust workspace member.
- **APIs:** additive Go package.
- **Dependencies:** `sciencekit_ffi` shared library; Go 1.22+.
- **Downstream:** exposes the library to Go services.
- **Acceptance criteria:** module builds, every family fits/predicts, errors map, finalizers
  close handles and tests pass.
