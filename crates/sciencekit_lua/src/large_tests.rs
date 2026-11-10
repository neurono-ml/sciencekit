//! Large-array tests (spec `lua-bindings`, acceptance: "runs with lots and
//! little data").
//!
//! The table conversion is element-by-element, so a large fixture proves the
//! copy path and the ABI calls scale without losing rectangularity. Run with
//! `cargo test -p sciencekit_lua --release` for the intended timing.

use crate::testing::sk_lua_test_environment;

/// A 2000 x 20 table fits and predicts back to the requested tolerance.
#[test]
fn large_table_fit_and_predict_round_trip() {
    let lua = sk_lua_test_environment();
    lua.globals().set("row_count", 2_000).expect("rows");
    lua.globals().set("column_count", 20).expect("columns");
    let worst_error: f64 = lua
        .load(
            r#"
            local sciencekit = require("sciencekit")
            local rows, columns = row_count, column_count
            local features, targets = {}, {}
            for row = 1, rows do
                local values = {}
                for column = 1, columns do
                    values[column] = ((row * column) % 97) * 0.25 + column
                end
                features[row] = values
                targets[row] = 2.5 * values[1] - 1.5 * values[2] + 0.25
            end
            local model = sciencekit.linear_regression():fit(features, targets)
            local predictions = model:predict(features)
            assert(#predictions == rows)
            local worst = 0.0
            for row = 1, rows do
                local delta = math.abs(predictions[row][1] - targets[row])
                if delta > worst then worst = delta end
            end
            return worst
        "#,
        )
        .eval()
        .expect("the large fixture fits and predicts");
    assert!(worst_error < 1.0e-8, "worst error was {worst_error}");
}

/// A large scaler transform keeps one output row per input row.
#[test]
fn large_table_transform_keeps_the_row_count() {
    let lua = sk_lua_test_environment();
    let rows: usize = lua
        .load(
            r#"
            local sciencekit = require("sciencekit")
            local features = {}
            for row = 1, 1000 do
                features[row] = {row * 0.5, row * 0.25, -row * 0.125}
            end
            local model = sciencekit.standard_scaler():fit(features)
            local transformed = model:transform(features)
            assert(#transformed[1] == 3)
            return #transformed
        "#,
        )
        .eval()
        .expect("the large transform succeeds");
    assert_eq!(rows, 1_000);
}
