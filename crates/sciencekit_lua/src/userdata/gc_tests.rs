//! Garbage-collection and model-lifecycle tests (spec `lua-bindings`,
//! requirement "Errors and lifecycle").

use crate::array::{SKLuaMatrix, SKLuaVector};
use crate::engine::SKLuaBuilderHandle;
use crate::engine::SKLuaEstimatorFamily;
use crate::lifecycle::sk_lua_live_handles;
use crate::testing::sk_lua_test_environment;
use crate::userdata::{sk_lua_builder_userdata, sk_lua_model_userdata};

/// The garbage collector frees builder and model handles exactly once.
#[test]
fn userdata_gc_frees_each_handle_exactly_once() {
    let lua = sk_lua_test_environment();
    let baseline = sk_lua_live_handles();
    lua.load(
        r#"
        local sciencekit = require("sciencekit")
        for index = 1, 16 do
            local builder = sciencekit.linear_regression()
            builder:include_intercept(index % 2 == 0)
        end
    "#,
    )
    .exec()
    .expect("builders are created and dropped");
    lua.gc_collect().expect("the collector runs");
    lua.gc_collect().expect("the collector runs again");
    assert_eq!(sk_lua_live_handles(), baseline);

    let features = SKLuaMatrix::new(4, 2, vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 2.0, 3.0]);
    let targets = SKLuaVector::new(vec![1.0, 3.0, 4.0, 14.0]);
    let model = {
        let builder =
            SKLuaBuilderHandle::create(SKLuaEstimatorFamily::LinearRegression).expect("builder");
        builder.fit(&features, Some(&targets)).expect("fit")
    };
    assert_eq!(sk_lua_live_handles(), baseline + 1);
    let userdata = sk_lua_model_userdata(&lua, model, SKLuaEstimatorFamily::LinearRegression)
        .expect("model userdata");
    lua.globals().set("gc_model", userdata).expect("global");
    lua.load("gc_model = nil")
        .exec()
        .expect("the global clears");
    lua.gc_collect().expect("the collector runs");
    lua.gc_collect().expect("the collector runs again");
    assert_eq!(sk_lua_live_handles(), baseline);
}

/// Closing a model twice is a no-op and marks it closed.
#[test]
fn userdata_model_close_is_idempotent() {
    let lua = sk_lua_test_environment();
    let features = SKLuaMatrix::new(2, 1, vec![1.0, 2.0]);
    let targets = SKLuaVector::new(vec![2.0, 4.0]);
    let model = {
        let builder = SKLuaBuilderHandle::create(SKLuaEstimatorFamily::LinearRegression).unwrap();
        builder.fit(&features, Some(&targets)).unwrap()
    };
    let userdata = sk_lua_model_userdata(&lua, model, SKLuaEstimatorFamily::LinearRegression)
        .expect("model userdata");
    lua.globals().set("model", userdata).expect("global");
    let closed: bool = lua
        .load(
            r#"
            model:close()
            model:close()
            return model.closed
        "#,
        )
        .eval()
        .expect("close is idempotent");
    assert!(closed);
}

/// The live-handle counter tracks creation and collection.
#[test]
fn userdata_live_handle_counter_tracks_creation() {
    let lua = sk_lua_test_environment();
    let baseline = sk_lua_live_handles();
    let builder = sk_lua_builder_userdata(&lua, SKLuaEstimatorFamily::StandardScaler).unwrap();
    assert_eq!(sk_lua_live_handles(), baseline + 1);
    drop(builder);
    lua.gc_collect().ok();
    lua.gc_collect().ok();
    assert_eq!(sk_lua_live_handles(), baseline);
}
