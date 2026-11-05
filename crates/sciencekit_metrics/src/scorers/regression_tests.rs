//! TDD tests for the regression scorer adapters (spec `evaluation-metrics`).

use ndarray::{Array1, Array2, array};
use sciencekit_common::{
    SKDataView, SKError, SKFloat, SKRegressorPredictor, SKSupervisedScorer, SKTargetView,
};

use super::{SKMeanAbsoluteErrorScorer, SKMeanSquaredErrorScorer, SKR2Scorer};

/// A regressor returning fixed predictions, ignoring the features.
struct FixedRegressor<F> {
    predictions: Vec<F>,
}

impl<F: SKFloat> SKRegressorPredictor<F> for FixedRegressor<F> {
    type Error = SKError;

    fn predict<'a, X>(&self, _features: X) -> Result<Array1<F>, SKError>
    where
        X: TryInto<SKDataView<'a, F>, Error = SKError>,
    {
        Ok(Array1::from_vec(self.predictions.clone()))
    }
}

/// A regressor whose inference must never run in the pure form.
struct PanickingRegressor;

impl<F: SKFloat> SKRegressorPredictor<F> for PanickingRegressor {
    type Error = SKError;

    fn predict<'a, X>(&self, _features: X) -> Result<Array1<F>, SKError>
    where
        X: TryInto<SKDataView<'a, F>, Error = SKError>,
    {
        panic!("the pure form must not run inference")
    }
}

/// The pure form scores stored predictions without running inference.
#[test]
fn pure_form_scores_without_inference() {
    let scorer = SKMeanSquaredErrorScorer;
    let targets = array![1.0_f64, 2.0, 3.0];
    let truth = SKTargetView::<f64>::Continuous(targets.view());
    let predictions = array![1.0_f64, 2.0, 4.0];
    let value = <SKMeanSquaredErrorScorer as SKSupervisedScorer<f64, PanickingRegressor>>::
        score_from_predictions(&scorer, truth, predictions.view())
        .expect("valid input");
    assert!((value - 1.0 / 3.0).abs() < 1e-12);
}

/// The convenient form equals the pure form applied to the model's predictions.
#[test]
fn convenient_form_equals_pure_form() {
    let scorer = SKR2Scorer;
    let model = FixedRegressor {
        predictions: vec![1.0_f64, 2.0, 4.0],
    };
    let features = Array2::<f64>::zeros((3, 2));
    let targets = array![1.0_f64, 2.0, 3.0];
    let truth = SKTargetView::<f64>::Continuous(targets.view());
    let convenient = scorer
        .score(&model, SKDataView::Dense(features.view()), truth)
        .expect("valid input");
    let pure = <SKR2Scorer as SKSupervisedScorer<f64, FixedRegressor<f64>>>::score_from_predictions(
        &scorer,
        truth,
        array![1.0_f64, 2.0, 4.0].view(),
    )
    .expect("valid input");
    assert!((convenient - pure).abs() < 1e-12);
    assert!((convenient - 0.5).abs() < 1e-12);
}

/// Every regression adapter produces its documented metric.
#[test]
fn regression_adapters_produce_their_metrics() {
    let targets = array![1.0_f64, 2.0, 3.0];
    let truth = SKTargetView::<f64>::Continuous(targets.view());
    let predictions = array![1.0_f64, 2.0, 4.0];

    let mae = <SKMeanAbsoluteErrorScorer as SKSupervisedScorer<f64, PanickingRegressor>>::
        score_from_predictions(&SKMeanAbsoluteErrorScorer, truth, predictions.view())
        .expect("valid input");
    assert!((mae - 1.0 / 3.0).abs() < 1e-12);

    let r2 = <SKR2Scorer as SKSupervisedScorer<f64, PanickingRegressor>>::score_from_predictions(
        &SKR2Scorer,
        truth,
        predictions.view(),
    )
    .expect("valid input");
    assert!((r2 - 0.5).abs() < 1e-12);
}

/// Nominal targets are rejected with guidance to encode them first.
#[test]
fn nominal_targets_are_rejected() {
    let scorer = SKMeanSquaredErrorScorer;
    let truth = SKTargetView::<f64>::Nominal(&["a", "b"]);
    let predictions = array![0.0_f64, 0.0];
    let error = <SKMeanSquaredErrorScorer as SKSupervisedScorer<f64, PanickingRegressor>>::
        score_from_predictions(&scorer, truth, predictions.view())
        .expect_err("nominal targets are not continuous");
    assert!(matches!(
        error,
        SKError::UnsupportedRepresentation {
            representation: "nominal",
            ..
        }
    ));
}
