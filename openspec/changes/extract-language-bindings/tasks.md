# Tasks

## 1. The Go repository

- [x] 1.1 Create `neurono-ml/sciencekit-go` with the module at its root, renamed to
  `github.com/neurono-ml/sciencekit-go`, plus a README, an Apache-2.0 LICENSE and a
  `.gitignore`; verify by building the module.
- [x] 1.2 Point the cgo directives at the vendored `include/sciencekit.h` and at `lib/`, and
  add `scripts/build-native-library.sh` and `scripts/sync-header.sh`; verify that
  `CGO_CFLAGS`, `CGO_LDFLAGS` and `LD_LIBRARY_PATH` override both.
- [x] 1.3 Add the CI workflow — `gofmt -l`, `go vet`, `go test`, `go test -race` and
  `GOEXPERIMENT=cgocheck2 go test` against `sciencekit_ffi` built from this repository —
  and verify every command locally.

## 2. The Lua repository

- [x] 2.1 Create `neurono-ml/sciencekit-lua` as a standalone Cargo project depending on
  `sciencekit_ffi` through a git dependency on this repository, keeping Lua 5.4 embedded
  (`mlua` `lua54` + `vendored`); verify with `cargo build --release`.
- [x] 2.2 Move the rockspec and the Lua suite (`tests/run.lua`) into the new repository;
  verify with `SCIENCEKIT_LUA_LIB=target/release/libsciencekit_lua.so lua5.4 tests/run.lua`.
- [x] 2.3 Add the CI workflow — build the module, run the Rust suite and the Lua suite —
  and verify every command locally.

## 3. The removal from this repository

- [x] 3.1 Delete `go/`, `go.work`, `crates/sciencekit_lua/`,
  `.github/workflows/go-bindings.yml` and `scripts/build-go-library.sh`, and remove
  `sciencekit_lua` from the workspace members; verify with `cargo metadata`.
- [x] 3.2 Update `CHANGELOG.md` with a "Moved" entry and `docs/src/contribute.md` with the
  new binding locations; verify with `mdbook build docs`.
- [x] 3.3 Update the `go-bindings` and `lua-bindings` design, proposal and task artifacts to
  name the new repositories and file layouts; verify with `openspec validate`.
- [x] 3.4 Run the gates — `cargo fmt --all --check`,
  `cargo clippy --workspace --all-targets -- -D warnings` and `cargo test --workspace` — and
  fix every failure before pushing.

## 4. Independent review

- [ ] 4.1 Independent review by a separate agent validating that the extraction preserves
  the `go-bindings` and `lua-bindings` behavior and that the Rust workspace is unaffected;
  address every finding and re-run the gates.
