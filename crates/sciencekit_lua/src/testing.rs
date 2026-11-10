//! Shared test setup: an embedded Lua 5.4 VM with the module registered.
//!
//! Tests never shell out to a `lua` executable. They build the module table
//! and publish it through `package.loaded`, exactly as `require` would, so the
//! same code path is exercised inside the vendored interpreter.

use mlua::{Lua, Table};

/// A fresh embedded VM where `require("sciencekit")` returns the module.
pub(crate) fn sk_lua_test_environment() -> Lua {
    let lua = Lua::new();
    let module = crate::sk_lua_build_module(&lua).expect("the module table builds");
    let package: Table = lua
        .globals()
        .get("package")
        .expect("the package library is loaded");
    let loaded: Table = package.get("loaded").expect("package.loaded exists");
    loaded
        .set("sciencekit", module)
        .expect("the module registers in package.loaded");
    lua
}
