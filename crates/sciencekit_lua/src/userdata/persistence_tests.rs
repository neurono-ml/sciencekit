//! Persistence, closed-handle and target-shape tests (spec `lua-bindings`,
//! requirements "Builders and models" and "Errors and lifecycle").

use crate::testing::sk_lua_test_environment;

/// A model saved to disk loads back and predicts identically.
#[test]
fn operations_save_and_load_round_trip() {
    let lua = sk_lua_test_environment();
    let path = std::env::temp_dir().join(format!(
        "sciencekit-lua-{}-round-trip.model",
        std::process::id()
    ));
    lua.globals()
        .set("model_path", path.to_string_lossy().to_string())
        .expect("the path global is writable");
    let prediction: f64 = lua
        .load(
            r#"
            local sciencekit = require("sciencekit")
            local features = {{0.0, 0.0}, {1.0, 0.0}, {0.0, 1.0}, {2.0, 3.0}}
            local targets = {1.0, 3.0, 4.0, 14.0}
            local model = sciencekit.linear_regression():fit(features, targets)
            model:save(model_path)
            local loaded = sciencekit.load(model_path)
            assert(loaded.family == "linear-regression")
            return loaded:predict({{2.0, 3.0}})[1][1]
        "#,
        )
        .eval()
        .expect("save and load succeed");
    let _ = std::fs::remove_file(&path);
    assert!(
        (prediction - 14.0).abs() < 1.0e-8,
        "loaded prediction {prediction}"
    );
}

/// A closed model rejects every operation with the invalid-handle status.
#[test]
fn operations_closed_model_rejects_predict() {
    let lua = sk_lua_test_environment();
    let message: String = lua
        .load(
            r#"
            local sciencekit = require("sciencekit")
            local features = {{1.0}, {2.0}, {3.0}}
            local model = sciencekit.linear_regression():fit(features, {2.0, 4.0, 6.0})
            model:close()
            local ok, err = pcall(function() model:predict(features) end)
            assert(not ok)
            return tostring(err)
        "#,
        )
        .eval()
        .expect("predict after close fails");
    assert!(message.contains("invalid-handle"), "unexpected: {message}");
}

/// Targets may arrive as a column of one-element rows.
#[test]
fn operations_targets_accept_a_column_of_rows() {
    let lua = sk_lua_test_environment();
    let score: f64 = lua
        .load(
            r#"
            local sciencekit = require("sciencekit")
            local features = {{1.0}, {2.0}, {3.0}}
            local targets = {{2.0}, {4.0}, {6.0}}
            local model = sciencekit.linear_regression():fit(features, targets)
            return model:score(features, targets)
        "#,
        )
        .eval()
        .expect("column targets are accepted");
    assert!((score - 1.0).abs() < 1.0e-8);
}
