//! Userdata lifecycle tests (spec `lua-bindings`, requirement "Errors and
//! lifecycle"): close is idempotent and the garbage collector releases each
//! handle exactly once.

use crate::lifecycle::sk_lua_live_handles;
use crate::testing::sk_lua_test_environment;

/// Constructors return userdata for each exposed family, with the documented
/// fields.
#[test]
fn userdata_constructors_expose_family_and_dtype() {
    let lua = sk_lua_test_environment();
    let (family, dtype, closed, scaler_family): (String, String, bool, String) = lua
        .load(
            r#"
            local sciencekit = require("sciencekit")
            local regression = sciencekit.linear_regression()
            local scaler = sciencekit.standard_scaler()
            return regression.family, regression.dtype, regression.closed, scaler.family
        "#,
        )
        .eval()
        .expect("constructors work");
    assert_eq!(family, "linear-regression");
    assert_eq!(dtype, "f64");
    assert!(!closed);
    assert_eq!(scaler_family, "standard-scaler");
}

/// Setters return the same userdata, so calls chain.
#[test]
fn userdata_setters_chain_on_the_same_handle() {
    let lua = sk_lua_test_environment();
    let chained: bool = lua
        .load(
            r#"
            local sciencekit = require("sciencekit")
            local regression = sciencekit.linear_regression()
            assert(regression:include_intercept(false) == regression)
            assert(regression:backend("faer") == regression)
            assert(regression:backend(0) == regression)
            local scaler = sciencekit.standard_scaler()
            assert(scaler:with_mean(true) == scaler)
            assert(scaler:with_std(false) == scaler)
            return true
        "#,
        )
        .eval()
        .expect("setters chain");
    assert!(chained);
}

/// A setter that belongs to another family reports an unsupported status.
#[test]
fn userdata_setter_rejects_the_wrong_family() {
    let lua = sk_lua_test_environment();
    let message: String = lua
        .load(
            r#"
            local sciencekit = require("sciencekit")
            local regression = sciencekit.linear_regression()
            local ok, err = pcall(function() regression:with_mean(true) end)
            assert(not ok)
            return tostring(err)
        "#,
        )
        .eval()
        .expect("pcall captures the error");
    assert!(
        message.contains("unsupported"),
        "unexpected error: {message}"
    );
}

/// Unknown backend names and codes are rejected before reaching the ABI.
#[test]
fn userdata_backend_rejects_unknown_names() {
    let lua = sk_lua_test_environment();
    for script in [
        r#"local b = require("sciencekit").linear_regression(); b:backend("blas")"#,
        r#"local b = require("sciencekit").linear_regression(); b:backend(9)"#,
        r#"local b = require("sciencekit").linear_regression(); b:backend(true)"#,
    ] {
        let error = lua.load(script).exec().expect_err("the backend is invalid");
        assert!(error.to_string().contains("invalid-argument"));
    }
}

/// `close` releases the handle and is idempotent; a closed builder rejects
/// further setters with the invalid-handle status.
#[test]
fn userdata_close_is_idempotent_and_rejects_further_use() {
    let lua = sk_lua_test_environment();
    let message: String = lua
        .load(
            r#"
            local sciencekit = require("sciencekit")
            local builder = sciencekit.linear_regression()
            assert(builder:close() == nil)
            assert(builder:close() == nil)
            assert(builder.closed)
            local ok, err = pcall(function() builder:include_intercept(true) end)
            assert(not ok)
            return tostring(err)
        "#,
        )
        .eval()
        .expect("close is idempotent");
    assert!(
        message.contains("invalid-handle"),
        "unexpected error: {message}"
    );
}

/// A to-be-closed builder releases its handle at scope exit.
#[test]
fn userdata_close_metamethod_releases_the_handle() {
    let lua = sk_lua_test_environment();
    let baseline = sk_lua_live_handles();
    lua.load(
        r#"
        local sciencekit = require("sciencekit")
        do
            local builder <close> = sciencekit.standard_scaler()
            assert(not builder.closed)
        end
    "#,
    )
    .exec()
    .expect("to-be-closed variables work");
    assert_eq!(sk_lua_live_handles(), baseline);
}
