# Proposal

## Why

Lua is a common embedding language for data pipelines and game/tooling environments. It can
use the shared C ABI, but a thin Lua module with idiomatic tables and userdata is far more
usable than raw `ffi` calls. `mlua 0.10` (MSRV 1.79) provides safe Lua 5.4 integration.

## What Changes

- New `sciencekit_lua` crate building a `sciencekit` Lua module over the C ABI (or `mlua`
  directly where it is simpler).
- Lua userdata for builders and fitted models; constructors, setters, `fit`, `predict`,
  `transform`, `score`, `save`, `load`.
- Numeric matrices as Lua tables of rows; explicit conversion to/from `SKFfiArray`/ndarray.
- Error mapping to Lua errors with the original message and status code.
- Lifecycle: userdata `__gc` frees handles; explicit `close` where supported.
- A luarocks rockspec and a Lua 5.4 test suite (no feature gate: the module is the crate's only product).
- Zero-copy is not promised across Lua tables; documented copy semantics.
- Explicit non-goals: LuaJIT FFI, async, GPU-specific APIs, Lua 5.1–5.3 compatibility.

## Capabilities

### New Capabilities

- `lua-bindings`: module loading, userdata lifecycle, array conversion, operations, errors
  and packaging.

### Modified Capabilities

None.

## Impact

- **Code:** new `crates/sciencekit_lua` workspace member plus Lua package files.
- **APIs:** additive Lua module.
- **Dependencies:** `mlua 0.10` with Lua 5.4; optional.
- **Downstream:** exposes the library to Lua tooling.
- **Acceptance criteria:** module loads, every family fits/predicts on Lua tables, errors
  map, GC frees handles and tests pass on Lua 5.4.
