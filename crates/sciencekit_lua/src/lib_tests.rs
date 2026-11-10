//! Module loading tests (spec `lua-bindings`, requirement "Module loading").

use mlua::{Table, Value};

use crate::testing::sk_lua_test_environment;

/// `require("sciencekit")` resolves to the module table inside the embedded
/// Lua 5.4 VM.
#[test]
fn module_loads_through_require() {
    let lua = sk_lua_test_environment();
    let module: Table = lua
        .load(r#"return require("sciencekit")"#)
        .eval()
        .expect("require succeeds");
    assert_eq!(
        module.get::<String>("version").expect("version"),
        env!("CARGO_PKG_VERSION")
    );
}

/// The module reports the ABI version and feature bitmask it was built with.
#[test]
fn module_reports_abi_version_and_features() {
    let lua = sk_lua_test_environment();
    let (major, minor, patch, features): (i32, i32, i32, i64) = lua
        .load(
            r#"
            local sciencekit = require("sciencekit")
            local version = sciencekit.abi_version()
            return version.major, version.minor, version.patch, sciencekit.features()
        "#,
        )
        .eval()
        .expect("version query succeeds");
    assert_eq!(major, sciencekit_ffi::SK_FFI_ABI_MAJOR);
    assert_eq!(minor, sciencekit_ffi::SK_FFI_ABI_MINOR);
    assert_eq!(patch, sciencekit_ffi::SK_FFI_ABI_PATCH);
    assert_eq!(features, sciencekit_ffi::sk_ffi_features() as i64);
}

/// The module is a table and never a callable proxy.
#[test]
fn module_value_is_a_table() {
    let lua = sk_lua_test_environment();
    let is_table: bool = lua
        .load(r#"return type(require("sciencekit")) == "table""#)
        .eval()
        .expect("type check succeeds");
    assert!(is_table);
    let value: Value = lua
        .load(r#"return require("sciencekit")"#)
        .eval()
        .expect("module value");
    assert!(matches!(value, Value::Table(_)));
}
