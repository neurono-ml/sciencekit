# sciencekit_lua

Lua 5.4 bindings for the `sciencekit` machine-learning library (spec
`lua-bindings`). The crate builds a native Lua module named `sciencekit`,
loaded with `require("sciencekit")`.

Lua 5.4 is **embedded**: the `mlua` dependency enables `lua54` + `vendored`,
so the interpreter is compiled from source into the module. No system
`liblua`, `pkg-config` file or runtime interpreter is required to build the
crate or run its Rust tests.

```lua
local sciencekit = require("sciencekit")

local features = { {0.0, 0.0}, {1.0, 0.0}, {0.0, 1.0}, {2.0, 3.0} }
local targets = { 1.0, 3.0, 4.0, 14.0 }

local model = sciencekit.linear_regression()
  :include_intercept(true)
  :backend("faer")
  :fit(features, targets)

local predictions = model:predict(features)   -- table of one-element rows
local r_squared = model:score(features, targets)
model:save("/tmp/model.state")

local scaler = sciencekit.standard_scaler():with_mean(true):with_std(true)
local transformed = scaler:fit(features):transform(features)
```

## Scope

- Estimator families: the ones the C ABI carries — linear regression
  (fit/predict/score/save/load) and the standard scaler (fit/transform).
- Scalars: `f64` only. Lua 5.4 numbers are IEEE-754 doubles, so an `f32`
  descriptor would add rounding without saving memory.
- **Copy semantics:** a Lua table is not a numeric buffer. Every input matrix
  or vector is copied into a contiguous `f64` buffer, and every output is a
  fresh table of rows. No zero-copy path is promised across Lua values.
- Non-goals: LuaJIT, async, GPU-specific APIs and Lua 5.1–5.3.

## Errors

Failures are Lua runtime errors headed by the stable status name and code:

```text
sciencekit shape (2): feature row 2 has 1 values, expected 2; a matrix must be rectangular
sciencekit invalid-argument (3): value at row 1, column 2 must be a number, found string
sciencekit invalid-handle (1): the model was already closed
sciencekit unsupported (4): ...
```

Tables are validated before any ABI call, so ragged matrices and non-numeric
cells never reach the library.

## Lifecycle

Builders and models own an opaque handle released exactly once by any of:

- `model:close()` / `builder:close()` (idempotent);
- the Lua 5.4 `__close` metamethod, via `local model <close> = ...`;
- the garbage collector, which drops the Rust payload.

`sciencekit.live_handles()` reports the number of handles live on the calling
thread and is useful to assert that a scope leaked nothing.

## Building and testing

Embedded path (default, no Lua installation needed):

```sh
cargo test -p sciencekit_lua
cargo build -p sciencekit_lua --release
```

System-interpreter smoke run (a `lua5.4` binary is needed):

```sh
cargo build -p sciencekit_lua --release
SCIENCEKIT_LUA_LIB=target/release/libsciencekit_lua.so lua5.4 crates/sciencekit_lua/tests/run.lua
```

Packaging through LuaRocks:

```sh
cd crates/sciencekit_lua
luarocks --lua-version=5.4 make sciencekit-0.5.0-1.rockspec
lua5.4 tests/run.lua    # resolves require("sciencekit") from the installed rock
```

The rockspec is a LuaRocks 3.0 `command` build: it runs `cargo build` and
installs the shared object as `sciencekit.so`. LuaRocks itself still probes
the Lua 5.4 development headers during configuration even though the module
embeds Lua, so the packaging host needs a `liblua5.4-dev` package (or an
equivalent `LUA_INCDIR`). The embedded and system-interpreter paths above do
not need it.

## Design

The module calls the stable `sciencekit_ffi` entry points in process
(`sciencekit_ffi` is linked as an `rlib`) instead of duplicating estimator
configuration, fitting, validation or persistence. Lua code therefore
exercises exactly the surface a C consumer sees; this crate only converts
tables, maps statuses and manages userdata lifetime.
