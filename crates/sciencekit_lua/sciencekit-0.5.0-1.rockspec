rockspec_format = "3.0"

package = "sciencekit"
version = "0.5.0-1"

source = {
  url = "git+https://github.com/neurono-ml/sciencekit.git",
  tag = "v0.5.0",
}

description = {
  summary = "Lua 5.4 bindings for the sciencekit machine-learning library",
  detailed = [[
    A native Lua 5.4 module that exposes the sciencekit C ABI with idiomatic
    userdata: chainable builders, fitted models with predict/transform/score,
    persistence, structured errors and table/array conversion. Lua 5.4 is
    embedded and compiled from source, so no system liblua is required.
  ]],
  homepage = "https://github.com/neurono-ml/sciencekit",
  license = "Apache-2.0",
}

dependencies = {
  "lua >= 5.4, < 5.5",
}

build = {
  -- The module is a Rust cdylib. LuaRocks substitutes $(LIBDIR) below with
  -- the rock's library directory; the built shared object is installed as
  -- sciencekit.so so require("sciencekit") finds it.
  type = "command",
  build_command = "cargo build --release -p sciencekit_lua",
  install_command = "mkdir -p \"$(LIBDIR)\" && cp target/release/libsciencekit_lua.so \"$(LIBDIR)/sciencekit.so\"",
}

test = {
  type = "command",
  command = "SCIENCEKIT_LUA_LIB=\"$(LIBDIR)/sciencekit.so\" lua5.4 tests/run.lua",
}
