//! `sciencekit_lua` — Lua 5.4 bindings for sciencekit (spec `lua-bindings`).
//!
//! The crate builds a native Lua module named `sciencekit` that is loaded with
//! `require("sciencekit")`. Lua 5.4 is **embedded**: the `mlua` dependency
//! enables `lua54` + `vendored`, so the interpreter is compiled from source
//! into the module and no system `liblua`, pkg-config file or runtime
//! interpreter is needed to build or to run the Rust tests.
//!
//! # Design choice: the C ABI, not duplicated math
//!
//! The module calls the stable entry points of `sciencekit_ffi` in process
//! (`sciencekit_ffi` is linked as an `rlib` for this purpose) instead of
//! re-deriving the estimator configuration, fitting and persistence logic.
//! Every operation therefore exercises exactly the surface a C consumer sees:
//! the same opaque handles, the same validation and the same status codes.
//! The Lua layer only converts tables, maps statuses to Lua errors and manages
//! userdata lifetime.
//!
//! # Copy semantics
//!
//! Lua tables are not numeric buffers. Every matrix or vector crossing the
//! boundary is copied into a contiguous `f64` buffer and back; predictions and
//! transforms are fresh tables of rows. This version promises no zero-copy
//! path across Lua values.
//!
//! # Scope
//!
//! The Lua module exposes the estimator families the C ABI carries: linear
//! regression and the standard scaler. Lua 5.4 numbers are IEEE-754 doubles,
//! so the module models in `f64` only; `f32` descriptors are out of scope for
//! this change. LuaJIT, async and GPU-specific APIs are explicit non-goals.

#![deny(unsafe_op_in_unsafe_fn)]
#![warn(missing_docs)]

mod array;
mod engine;
mod error;
mod lifecycle;
mod userdata;

use mlua::{Lua, Result, Table};

use crate::engine::SKLuaEstimatorFamily;
use crate::engine::SKLuaModelHandle;
use crate::userdata::{sk_lua_builder_userdata, sk_lua_model_userdata};

/// Build the `sciencekit` module table.
///
/// Factored out of the native entry point so the crate's tests can register
/// the very same table in an embedded VM without loading a shared object.
pub(crate) fn sk_lua_build_module(lua: &Lua) -> Result<Table> {
    let exports = lua.create_table()?;
    exports.set("version", env!("CARGO_PKG_VERSION"))?;
    exports.set(
        "abi_version",
        lua.create_function(|lua, ()| {
            let table = lua.create_table()?;
            table.set("major", sciencekit_ffi::SK_FFI_ABI_MAJOR)?;
            table.set("minor", sciencekit_ffi::SK_FFI_ABI_MINOR)?;
            table.set("patch", sciencekit_ffi::SK_FFI_ABI_PATCH)?;
            Ok(table)
        })?,
    )?;
    exports.set(
        "features",
        lua.create_function(|_, ()| Ok(sciencekit_ffi::sk_ffi_features() as i64))?,
    )?;
    exports.set(
        "live_handles",
        lua.create_function(|_, ()| Ok(lifecycle::sk_lua_live_handles()))?,
    )?;
    exports.set(
        "linear_regression",
        lua.create_function(|lua, ()| {
            sk_lua_builder_userdata(lua, SKLuaEstimatorFamily::LinearRegression)
        })?,
    )?;
    exports.set(
        "standard_scaler",
        lua.create_function(|lua, ()| {
            sk_lua_builder_userdata(lua, SKLuaEstimatorFamily::StandardScaler)
        })?,
    )?;
    // The ABI's persistence scope is fitted f64 linear regression, so a
    // loaded model always reports the linear-regression family.
    exports.set(
        "load",
        lua.create_function(|lua, path: String| {
            let handle = SKLuaModelHandle::load(&path)?;
            sk_lua_model_userdata(lua, handle, SKLuaEstimatorFamily::LinearRegression)
        })?,
    )?;
    Ok(exports)
}

/// The native module entry point; Lua calls it as `luaopen_sciencekit`.
///
/// Written on [`mlua::Lua::entrypoint1`] rather than `#[mlua::lua_module]`
/// because `mlua`'s `module` feature and the `vendored` Lua build are
/// mutually exclusive. This entry point runs against the state the host
/// interpreter passes in, exactly like the macro's expansion.
///
/// # Safety
///
/// `state` must be a valid Lua state owned by the interpreter that loads the
/// module, and this function must only be reached through `require`.
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn luaopen_sciencekit(
    state: *mut mlua::lua_State,
) -> std::os::raw::c_int {
    unsafe { Lua::entrypoint1(state, sk_lua_build_module) }
}

#[cfg(test)]
#[path = "error_tests.rs"]
mod error_tests;

#[cfg(test)]
#[path = "concurrency_tests.rs"]
mod concurrency_tests;

#[cfg(test)]
#[path = "large_tests.rs"]
mod large_tests;

#[cfg(test)]
#[path = "lib_tests.rs"]
mod lib_tests;

#[cfg(test)]
#[path = "testing.rs"]
mod testing;
