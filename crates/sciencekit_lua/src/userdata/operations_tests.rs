//! Operation tests (spec `lua-bindings`, requirements "Builders and models"
//! and "Errors and lifecycle"): fit, predict, transform, score, save and load
//! through the public Lua surface.

use crate::testing::sk_lua_test_environment;

/// Linear regression fits, predicts and scores through Lua tables.
#[test]
fn operations_linear_regression_fits_predicts_and_scores() {
    let lua = sk_lua_test_environment();
    let (predictions, score): (mlua::Table, f64) = lua
        .load(
            r#"
            local sciencekit = require("sciencekit")
            local features = {{0.0, 0.0}, {1.0, 0.0}, {0.0, 1.0}, {2.0, 3.0}}
            local targets = {1.0, 3.0, 4.0, 14.0}
            local model = sciencekit.linear_regression():fit(features, targets)
            assert(model.family == "linear-regression")
            return model:predict(features), model:score(features, targets)
        "#,
        )
        .eval()
        .expect("fit, predict and score succeed");
    let expected = [1.0, 3.0, 4.0, 14.0];
    assert_eq!(predictions.raw_len(), expected.len());
    for (index, value) in expected.iter().enumerate() {
        let predicted: f64 = predictions
            .get::<mlua::Table>(index + 1)
            .expect("a row")
            .get(1)
            .expect("one column");
        assert!(
            (predicted - value).abs() < 1.0e-8,
            "prediction {index} was {predicted}, expected {value}"
        );
    }
    assert!((score - 1.0).abs() < 1.0e-8, "score was {score}");
}

/// The standard scaler transforms into a table of the same shape.
#[test]
fn operations_standard_scaler_transforms_with_the_expected_shape() {
    let lua = sk_lua_test_environment();
    let (rows, columns, first_row, last_row): (usize, usize, mlua::Table, mlua::Table) = lua
        .load(
            r#"
            local sciencekit = require("sciencekit")
            local features = {{0.0, 0.0}, {2.0, 2.0}, {4.0, 4.0}, {6.0, 6.0}}
            local model = sciencekit.standard_scaler():fit(features)
            local transformed = model:transform(features)
            return #transformed, #transformed[1], transformed[1], transformed[4]
        "#,
        )
        .eval()
        .expect("fit and transform succeed");
    assert_eq!(rows, 4);
    assert_eq!(columns, 2);
    let first: f64 = first_row.get(1).expect("first value");
    let last: f64 = last_row.get(1).expect("last value");
    assert!((first + last).abs() < 1.0e-9, "the column is centered");
    let second: f64 = first_row.get(2).expect("second value");
    assert!(
        (first - second).abs() < 1.0e-9,
        "both columns share a scale"
    );
}

/// Predictions always arrive as a table of one-element rows.
#[test]
fn operations_predictions_are_tables_of_rows() {
    let lua = sk_lua_test_environment();
    let (rows, columns, kind): (usize, usize, String) = lua
        .load(
            r#"
            local sciencekit = require("sciencekit")
            local features = {{1.0}, {2.0}, {3.0}}
            local targets = {2.0, 4.0, 6.0}
            local predictions = sciencekit.linear_regression():fit(features, targets):predict(features)
            return #predictions, #predictions[1], type(predictions[2][1])
        "#,
        )
        .eval()
        .expect("predict succeeds");
    assert_eq!(rows, 3);
    assert_eq!(columns, 1);
    assert_eq!(kind, "number");
}

/// A regression fit without targets is rejected by the ABI.
#[test]
fn operations_linear_regression_requires_targets() {
    let lua = sk_lua_test_environment();
    let message: String = lua
        .load(
            r#"
            local sciencekit = require("sciencekit")
            local ok, err = pcall(function()
                return sciencekit.linear_regression():fit({{1.0}, {2.0}})
            end)
            assert(not ok)
            return tostring(err)
        "#,
        )
        .eval()
        .expect("pcall captures the error");
    assert!(
        message.contains("invalid-argument"),
        "unexpected: {message}"
    );
}

/// Operations that belong to the other family report `unsupported`.
#[test]
fn operations_wrong_family_operations_are_unsupported() {
    let lua = sk_lua_test_environment();
    let messages: (String, String) = lua
        .load(
            r#"
            local sciencekit = require("sciencekit")
            local features = {{0.0, 0.0}, {1.0, 1.0}}
            local scaler = sciencekit.standard_scaler():fit(features)
            local ok_predict, predict_error = pcall(function() scaler:predict(features) end)
            assert(not ok_predict)
            local regression = sciencekit.linear_regression():fit(features, {1.0, 2.0})
            local ok_transform, transform_error = pcall(function() regression:transform(features) end)
            assert(not ok_transform)
            return tostring(predict_error), tostring(transform_error)
        "#,
        )
        .eval()
        .expect("unsupported operations fail");
    assert!(
        messages.0.contains("unsupported"),
        "unexpected: {}",
        messages.0
    );
    assert!(
        messages.1.contains("unsupported"),
        "unexpected: {}",
        messages.1
    );
}
