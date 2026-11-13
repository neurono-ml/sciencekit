//! Companion tests for `SKGradientBoostingRegressor` across `SquaredError`,
//! `AbsoluteError` and `Huber`, hand-computed updates and smooth responses
//! (task 5.2).

use ndarray::{Array1, Array2, array};
use sciencekit_common::{SKBackendKind, SKBuilder, SKRegressorPredictor, SKSupervisedFit};
use sciencekit_metrics::regression::sk_r2_score;

use super::{SKGradientBoostingRegressionLoss, SKGradientBoostingRegressor};

/// Scenario: Regression sums the stages (`init = 2.0`, one stage tree
/// predicting `0.5`, `learning_rate = 0.1` -> `2.05`), plus hand-computed
/// `AbsoluteError` and `Huber` one-stage steps.
#[test]
fn hand_computed_single_stage_updates_for_all_three_regression_losses() {
    let features = array![[0.0], [1.0], [10.0], [11.0]];
    let squared_targets = array![1.5_f64, 1.5, 2.5, 2.5];

    let squared_model = SKGradientBoostingRegressor::<f64>::build()
        .number_of_estimators(1)
        .maximum_depth(1)
        .learning_rate(0.1)
        .loss(SKGradientBoostingRegressionLoss::SquaredError)
        .build()
        .expect("builds")
        .fit(&features, squared_targets.view())
        .expect("fits");

    assert!((squared_model.initial_prediction() - 2.0).abs() < 1e-12);
    assert_eq!(squared_model.number_of_stages(), 1);
    let tree = squared_model.stage_tree(0).expect("stage 0");
    assert!((tree.leaf_value_for(|_| 10.5) - 0.5).abs() < 1e-12);
    let squared_predictions = squared_model
        .predict(array![[0.5], [10.5]].view())
        .expect("predicts");
    assert!((squared_predictions[0] - 1.95).abs() < 1e-12);
    assert!((squared_predictions[1] - 2.05).abs() < 1e-12);

    // AbsoluteError: targets [0, 0, 10, 10] -> init mean = 5.0;
    // sign gradients [-1, -1, +1, +1] with unit Hessian -> leaves [-1.0, +1.0];
    // learning_rate = 0.5 -> predictions [4.5, 5.5].
    let absolute_targets = array![0.0_f64, 0.0, 10.0, 10.0];
    let absolute_model = SKGradientBoostingRegressor::<f64>::build()
        .number_of_estimators(1)
        .maximum_depth(1)
        .learning_rate(0.5)
        .loss(SKGradientBoostingRegressionLoss::AbsoluteError)
        .build()
        .expect("builds")
        .fit(&features, absolute_targets.view())
        .expect("fits");
    let absolute_predictions = absolute_model
        .predict(array![[0.5], [10.5]].view())
        .expect("predicts");
    assert!((absolute_predictions[0] - 4.5).abs() < 1e-12);
    assert!((absolute_predictions[1] - 5.5).abs() < 1e-12);

    // Huber(delta = 1.0): targets [0, 0, 20, 20] -> init mean = 10.0;
    // residuals [-10, -10, +10, +10] clipped to [-1, -1, +1, +1] -> leaves [-1, +1];
    // learning_rate = 0.5 -> predictions [9.5, 10.5].
    let huber_targets = array![0.0_f64, 0.0, 20.0, 20.0];
    let huber_model = SKGradientBoostingRegressor::<f64>::build()
        .number_of_estimators(1)
        .maximum_depth(1)
        .learning_rate(0.5)
        .loss(SKGradientBoostingRegressionLoss::Huber)
        .huber_delta(1.0)
        .build()
        .expect("builds")
        .fit(&features, huber_targets.view())
        .expect("fits");
    let huber_predictions = huber_model
        .predict(array![[0.5], [10.5]].view())
        .expect("predicts");
    assert!((huber_predictions[0] - 9.5).abs() < 1e-12);
    assert!((huber_predictions[1] - 10.5).abs() < 1e-12);
}

/// Smooth nonlinear curve `y = sin(x) + 0.5 * x` is recovered accurately under
/// `SquaredError`, `AbsoluteError` and `Huber`, and integer targets promote
/// cleanly.
#[test]
fn smooth_response_recovered_under_every_regression_loss() {
    let sample_count = 80;
    let features = Array2::from_shape_fn((sample_count, 2), |(row, column)| {
        let coordinate = -3.0 + 6.0 * (row as f64 / (sample_count - 1) as f64);
        if column == 0 {
            coordinate
        } else {
            coordinate * coordinate
        }
    });
    let responses = Array1::from_shape_fn(sample_count, |row| {
        let coordinate = features[[row, 0]];
        coordinate.sin() + 0.5 * coordinate
    });

    for loss in [
        SKGradientBoostingRegressionLoss::SquaredError,
        SKGradientBoostingRegressionLoss::AbsoluteError,
        SKGradientBoostingRegressionLoss::Huber,
    ] {
        let model = SKGradientBoostingRegressor::<f64>::build()
            .number_of_estimators(80)
            .learning_rate(0.15)
            .maximum_depth(3)
            .loss(loss)
            .huber_delta(0.8)
            .backend(SKBackendKind::MatrixMultiply)
            .build()
            .expect("builds")
            .fit(&features, responses.view())
            .expect("fits");

        assert_eq!(model.loss(), loss);
        assert_eq!(model.backend(), SKBackendKind::MatrixMultiply);
        let predictions = model.predict(&features).expect("predicts");
        let r2 = sk_r2_score(responses.view(), predictions.view()).expect("r2");
        assert!(r2 > 0.92, "loss {loss:?} R² = {r2}");
    }

    let integer_targets = array![1_i64, 2, 10, 11];
    let promoted = SKGradientBoostingRegressor::<f64>::build()
        .number_of_estimators(10)
        .build()
        .expect("builds")
        .fit(&array![[0.0], [1.0], [5.0], [6.0]], integer_targets.view())
        .expect("integer targets promote");
    assert_eq!(promoted.stage_trees().len(), 10);
}
