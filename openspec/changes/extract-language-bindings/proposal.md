# Proposal

## Why

`sciencekit` is a Cargo workspace of Rust crates published to crates.io. Two language
bindings were implemented inside it, and each brought a second toolchain and a second
build system into a repository whose release pipeline publishes Rust crates:

- the Go binding, a `go/` module (module path `github.com/neurono-ml/sciencekit/go`) built
  with cgo over `sciencekit_ffi`, guarded by a root `go.work` file, a dedicated
  `.github/workflows/go-bindings.yml` workflow and a `scripts/build-go-library.sh` helper;
- the Lua binding, `crates/sciencekit_lua`, a Cargo workspace member built on `mlua`
  (`lua54` + `vendored`) that links `sciencekit_ffi` as an `rlib`.

Neither is a Rust library that ships on crates.io. The umbrella crate's `Cargo.toml` had
to `exclude` the Go module and the `go.work` file from the published tarball, and the Go
and Lua toolchains had to be installed in the development container for a Rust workspace
that never builds them.

The bindings are also consumed differently from the Rust crates: a Go program consumes the
module through `go get` and links the native library at runtime; a Lua program consumes the
rock through LuaRocks. Neither consumption model is served by living inside the Rust
workspace.

## What Changes

- The Go binding moves to its own repository, `neurono-ml/sciencekit-go`, as the module
  `github.com/neurono-ml/sciencekit-go`, at the repository root. The module path changes.
- The Lua binding moves to its own repository, `neurono-ml/sciencekit-lua`, as a standalone
  Cargo project (its own `[workspace]`) that depends on `sciencekit_ffi` through a git
  dependency on this repository.
- The `go/` module, the `go.work` file, the `crates/sciencekit_lua` crate, the
  `go-bindings` CI workflow and `scripts/build-go-library.sh` are removed from this
  repository; `sciencekit_lua` leaves the Cargo workspace members.
- The stable C ABI (`sciencekit_ffi`) stays in this repository. It is the shared contract
  both bindings consume, and its version is reported by the library and checked by each
  binding at initialization.
- The `go-bindings` and `lua-bindings` change artifacts are updated to name the new
  repositories and the new file layouts.
- No library behavior changes: the specifications of `go-bindings` and `lua-bindings`
  describe the bindings themselves and are unaffected by where the code lives.

## Capabilities

### New Capabilities

_None._ Tooling-only change (`skip_specs: true`): the `go-bindings` and `lua-bindings`
capabilities are unchanged; only the repository that hosts them changes.

### Modified Capabilities

_None._

## Impact

- **Code:** the Go and Lua sources leave this repository; `sciencekit_ffi` and the rest of
  the Rust workspace are untouched.
- **APIs:** the Go module path changes from `github.com/neurono-ml/sciencekit/go` to
  `github.com/neurono-ml/sciencekit-go`; the Lua module name (`require("sciencekit")`) and
  the C ABI are unchanged.
- **Dependencies:** the Lua crate resolves `sciencekit_ffi` from a git dependency instead
  of a sibling path; the Go binding resolves the native library at link time.
- **Downstream:** Go consumers update their import path; Lua consumers install the rock
  from its new source URL.
- **Acceptance:** the Rust workspace gates stay green (`cargo fmt --all --check`,
  `cargo clippy --workspace --all-targets -- -D warnings`, `cargo test --workspace`); the
  Go suite and the Lua suite pass in their new repositories.
