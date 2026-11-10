//! Error-mapping and double-free tests (spec `lua-bindings`, requirement
//! "Errors and lifecycle").
//!
//! Every failure is a Lua runtime error whose text starts with
//! `sciencekit <status-name> (<code>):`, so Lua code sees both the stable
//! status and the message the library produced.

use mlua::Error;
use sciencekit_ffi::SKFfiStatus;

use crate::array::{SKLuaMatrix, SKLuaVector};
use crate::engine::SKLuaBuilderHandle;
use crate::engine::SKLuaEstimatorFamily;
use crate::error::SKLuaError;
use crate::lifecycle::sk_lua_live_handles;
use crate::testing::sk_lua_test_environment;

/// Display renders the status name, its code and the message.
#[test]
fn errors_display_carries_the_status_and_code() {
    let error = SKLuaError::shape("a matrix must be rectangular");
    assert_eq!(
        error.to_string(),
        "sciencekit shape (2): a matrix must be rectangular"
    );
    let lua_error: Error = error.into();
    assert!(matches!(lua_error, Error::RuntimeError(_)));
}

/// Every status name maps to its stable code.
#[test]
fn errors_status_names_match_the_abi_codes() {
    let cases = [
        (SKFfiStatus::InvalidHandle, 1, "invalid-handle"),
        (SKFfiStatus::Shape, 2, "shape"),
        (SKFfiStatus::InvalidArgument, 3, "invalid-argument"),
        (SKFfiStatus::Unsupported, 4, "unsupported"),
        (SKFfiStatus::NotFitted, 5, "not-fitted"),
        (SKFfiStatus::Io, 6, "io"),
        (SKFfiStatus::Internal, 7, "internal"),
    ];
    for (status, code, name) in cases {
        let error = SKLuaError::new(status, "example");
        assert_eq!(error.status_code(), code);
        assert!(error.to_string().contains(name));
    }
}

/// The last-error message recorded by a failed ABI call is copied, not just
/// the status name.
#[test]
fn errors_copy_the_abi_last_error_message() {
    let pointer = sciencekit_ffi::sk_ffi_builder_create(99, 1);
    assert!(pointer.is_null());
    let error = SKLuaError::from_last_error(SKFfiStatus::InvalidArgument);
    assert!(
        error.message().contains("unknown estimator family"),
        "unexpected message: {}",
        error.message()
    );
}

/// A ragged table raises a shape error inside Lua before any ABI call.
#[test]
fn errors_ragged_table_reaches_lua_as_a_shape_error() {
    let lua = sk_lua_test_environment();
    let message: String = lua
        .load(
            r#"
            local sciencekit = require("sciencekit")
            local ok, err = pcall(function()
                sciencekit.linear_regression():fit({{1.0, 2.0}, {3.0}}, {1.0, 2.0})
            end)
            assert(not ok)
            return tostring(err)
        "#,
        )
        .eval()
        .expect("pcall captures the error");
    assert!(
        message.contains("sciencekit shape (2)"),
        "unexpected: {message}"
    );
    assert!(message.contains("rectangular"), "unexpected: {message}");
}

/// A non-numeric cell raises an invalid-argument error naming the position.
#[test]
fn errors_non_numeric_cell_reaches_lua_as_invalid_argument() {
    let lua = sk_lua_test_environment();
    let message: String = lua
        .load(
            r#"
            local sciencekit = require("sciencekit")
            local ok, err = pcall(function()
                sciencekit.standard_scaler():fit({{1.0, "two"}})
            end)
            assert(not ok)
            return tostring(err)
        "#,
        )
        .eval()
        .expect("pcall captures the error");
    assert!(
        message.contains("sciencekit invalid-argument (3)"),
        "unexpected: {message}"
    );
    assert!(message.contains("row 1, column 2"), "unexpected: {message}");
}

/// A library-side shape mismatch keeps the ABI's status and message.
#[test]
fn errors_library_shape_mismatch_reaches_lua() {
    let lua = sk_lua_test_environment();
    let message: String = lua
        .load(
            r#"
            local sciencekit = require("sciencekit")
            local features = {{1.0, 1.0}, {2.0, 2.0}}
            local model = sciencekit.linear_regression():fit(features, {1.0, 2.0})
            local ok, err = pcall(function() model:predict({{1.0}}) end)
            assert(not ok)
            return tostring(err)
        "#,
        )
        .eval()
        .expect("pcall captures the error");
    assert!(
        message.contains("sciencekit shape (2)"),
        "unexpected: {message}"
    );
}

/// Closing a builder twice releases exactly one handle.
#[test]
fn errors_double_close_frees_a_builder_once() {
    let baseline = sk_lua_live_handles();
    let mut builder =
        SKLuaBuilderHandle::create(SKLuaEstimatorFamily::LinearRegression).expect("builder");
    assert_eq!(sk_lua_live_handles(), baseline + 1);
    builder.close();
    builder.close();
    assert!(builder.is_closed());
    assert_eq!(sk_lua_live_handles(), baseline);
    // Dropping a closed handle must not free again.
    drop(builder);
    assert_eq!(sk_lua_live_handles(), baseline);
}

/// Closing a model twice releases exactly one handle.
#[test]
fn errors_double_close_frees_a_model_once() {
    let baseline = sk_lua_live_handles();
    let features = SKLuaMatrix::new(2, 1, vec![1.0, 2.0]);
    let targets = SKLuaVector::new(vec![2.0, 4.0]);
    let mut model = {
        let builder = SKLuaBuilderHandle::create(SKLuaEstimatorFamily::LinearRegression).unwrap();
        builder.fit(&features, Some(&targets)).expect("fit")
    };
    assert_eq!(sk_lua_live_handles(), baseline + 1);
    model.close();
    model.close();
    assert!(model.is_closed());
    assert_eq!(sk_lua_live_handles(), baseline);
    drop(model);
    assert_eq!(sk_lua_live_handles(), baseline);
}

/// Explicitly closing before the collector runs must not double free.
#[test]
fn errors_gc_after_close_does_not_double_free() {
    let lua = sk_lua_test_environment();
    let baseline = sk_lua_live_handles();
    lua.load(
        r#"
        local sciencekit = require("sciencekit")
        local features = {{1.0}, {2.0}, {3.0}}
        for _ = 1, 8 do
            local model = sciencekit.linear_regression():fit(features, {2.0, 4.0, 6.0})
            model:close()
        end
    "#,
    )
    .exec()
    .expect("closing inside the loop works");
    lua.gc_collect().expect("the collector runs");
    lua.gc_collect().expect("the collector runs again");
    assert_eq!(sk_lua_live_handles(), baseline);
}
