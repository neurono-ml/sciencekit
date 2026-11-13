# Design

## Context

See `proposal.md` — Why. `ffi-core` provides the stable C ABI; `mlua 0.10` (MSRV 1.79)
provides safe Lua userdata and module loading. Lua tables are not a zero-copy numeric
buffer, so conversion copies are explicit and documented. The Lua crate lives in its own
repository (`neurono-ml/sciencekit-lua`), outside this Cargo workspace.

## Goals / Non-Goals

**Goals:**
- A small, idiomatic Lua module.
- Safe handle lifecycle through `__gc`.
- Clear errors and array validation.

**Non-Goals:**
- LuaJIT FFI, async, GPU, zero-copy tables.
- Lua 5.1–5.3 compatibility.

## Decisions

### Decision 1 — `mlua` module

The crate exposes `a hand-written `luaopen_sciencekit` entry point (the `module` and `vendored` mlua features are mutually exclusive)`. Builders/models are
`UserData` structs holding an `SKFfiHandle` pointer; setters return `&mut Self` for
chaining. `__gc` calls `handle_free` exactly once.

### Decision 2 — Table conversion

A numeric matrix is a table of equal-length rows; conversion validates rectangularity and
constructs an `SKFfiArray`/ndarray copy. Outputs are tables of rows. A future buffer-backed
path can avoid copies; this version documents the copy.

### Decision 3 — Errors

`SKFfiStatus` maps to `mlua::Error::RuntimeError` with the status name and last-error
message. Panics are already contained by the C ABI.

### Decision 4 — Packaging

`sciencekit-<version>-1.rockspec` declares the native module build, Lua 5.4 dependency and
test command. Tests use `busted` or a plain Lua script.

### Decision 5 — File layout

The Lua crate lives in its own repository, `neurono-ml/sciencekit-lua`, as a standalone
Cargo project (its own `[workspace]`) at the repository root:

```
sciencekit-lua/
  Cargo.toml
  sciencekit-*.rockspec
  src/{lib.rs, userdata.rs, convert.rs, error.rs}
  tests/*.lua
```

The crate depends on `sciencekit_ffi` through a git dependency on the main repository, so
it is built and released independently of this workspace.

## Risks / Trade-offs

- **Table copies** → documented; large matrices are better served by another binding.
- **GC/finalizer ordering** → explicit `close` plus `__gc` idempotence.
- **Lua version** → only 5.4; documented.
- **mlua MSRV** → pin 0.10 (1.79).

## Migration Plan

Additive: new crate and Lua package in their own repository. No existing API changes.

## Open Questions

None blocking. A buffer-backed zero-copy path is future work.
