# Design

## Context

See `proposal.md` — Why. `ffi-core` defines the stable C ABI both bindings consume;
`go-bindings` and `lua-bindings` define their behavior. This change decides only where
the two bindings live.

## Goals / Non-Goals

**Goals:**
- One repository per toolchain.
- A clean published-crate surface for the Rust crates.
- An unchanged `sciencekit_ffi` C ABI that both bindings consume as a released artifact.

**Non-Goals:**
- Changing the binding behavior, the C ABI or the estimator surface.
- Publishing the bindings to a package registry (crates.io, the Go module proxy, LuaRocks).
- Moving the Python or JVM bindings, which have not been implemented yet.

## Decisions

### Decision 1 — One repository per binding

`neurono-ml/sciencekit-go` hosts the Go module at its repository root; the module path
becomes `github.com/neurono-ml/sciencekit-go`. `neurono-ml/sciencekit-lua` hosts the Lua
crate at its repository root as a standalone Cargo project.

Each repository carries its own README, LICENSE, `.gitignore`, examples, tests and CI
workflow, so it is usable without this repository checked out.

### Decision 2 — The C ABI stays here

`sciencekit_ffi` remains a member of this workspace and the single source of truth for the
ABI. It builds a `cdylib` (for the Go binding's link-time resolution), a `staticlib` and
an `rlib` (for the Lua binding's in-process calls).

The ABI version is reported by the library and checked by each binding at initialization:
the Go binding rejects a library whose ABI major differs from the one it implements, and
the Lua module exposes `abi_version` alongside the ABI status names.

### Decision 3 — The Go binding links a built shared library

The Go cgo directives resolve the ABI header from a vendored `include/sciencekit.h` and
the shared library from `lib/`. `CGO_CFLAGS`, `CGO_LDFLAGS` and `LD_LIBRARY_PATH` override
both, which is how a consumer points at a `sciencekit_ffi` built from a checkout of this
repository.

`scripts/build-native-library.sh` builds the library from a checkout of this repository and
installs it into `lib/`; `scripts/sync-header.sh` refreshes the vendored header. The
repository's CI runs both, then `gofmt -l`, `go vet`, `go test`, `go test -race` and
`GOEXPERIMENT=cgocheck2 go test`.

Vendoring the header keeps the module compilable before the native library exists; the CI
sync step keeps it from drifting.

### Decision 4 — The Lua binding depends on the C ABI through git

The Lua crate declares `sciencekit_ffi = { git = "https://github.com/neurono-ml/sciencekit",
branch = "main" }`. A released revision can be pinned by replacing `branch` with `rev`.

Lua 5.4 stays embedded (`mlua` with `lua54` + `vendored`), so neither a system `liblua` nor
pkg-config is needed to build the module. The rockspec, the Lua suite (`tests/run.lua`) and
the CI workflow that builds the module and runs both suites come along.

A git dependency (rather than a crates.io dependency) is used because `sciencekit_ffi` is
not published: it is consumed as a shared C ABI and as an `rlib`, and its release cadence is
tied to this repository.

## Risks / Trade-offs

- **The C ABI becomes a cross-repository contract** → the ABI version is checked at
  initialization by both bindings, and the header is vendored with a sync script.
- **The Go module path changes** → a breaking change for Go consumers, unavoidable when the
  module leaves a subdirectory of another repository.
- **The Lua crate resolves a git dependency** → a build fetches this repository; the
  version can be pinned with `rev`.
- **Two more repositories to maintain** → each carries its own CI, so the bindings are not
  covered by this repository's `cargo` gates any more.

## Migration Plan

1. Create `neurono-ml/sciencekit-go` and `neurono-ml/sciencekit-lua`; copy the sources and
   adapt the module path, the cgo directives, the Cargo manifest, the rockspec, the READMEs
   and the CI workflows.
2. Verify each repository builds and tests against the `sciencekit_ffi` built from this
   repository.
3. Remove the bindings from this repository and merge that removal.

Go consumers update their import path to `github.com/neurono-ml/sciencekit-go`. Lua
consumers install the rock from `https://github.com/neurono-ml/sciencekit-lua`.

## Open Questions

None blocking. Publishing the bindings to a package registry is future work.
