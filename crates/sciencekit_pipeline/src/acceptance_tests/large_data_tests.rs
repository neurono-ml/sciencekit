//! Large-data acceptance: the chain fits and predicts over many rows without
//! changing its contract.

use sciencekit_common::{SKBuilder, SKClassifierPredictor, SKRegressorPredictor};
use sciencekit_impute::simple_imputer::SKSimpleImputer;
use sciencekit_linear_model::logistic_regression::SKLogisticRegression;
use sciencekit_metrics::classification::sk_accuracy_score;
use sciencekit_preprocessing::standard_scaler::core_implementation::SKStandardScaler;

use super::fixtures::{large_features, large_labels};
use crate::SKPipeline;

#[test]
fn a_large_classifier_pipeline_fits_and_predicts() {
    let features = large_features(400, 6);
    let labels = large_labels(400);
    let fitted = SKPipeline::start::<f64>()
        .then(SKStandardScaler::build().build().unwrap())
        .then(SKSimpleImputer::build().build().unwrap())
        .finalize_classifier(SKLogisticRegression::build().build().unwrap())
        .unwrap()
        .fit(&features, labels.view())
        .unwrap();
    let predictions = fitted.predict_labels(&features).unwrap();
    assert_eq!(predictions.len(), 400);
    let truth: Vec<i64> = labels.to_vec();
    let predicted: Vec<i64> = predictions.to_vec();
    let accuracy: f64 = sk_accuracy_score(&truth, &predicted).unwrap();
    assert!(accuracy > 0.5, "accuracy was {accuracy}");
}

#[test]
fn a_large_regressor_pipeline_fits_and_predicts() {
    let features = large_features(300, 4);
    let targets = large_labels(300).mapv(|label| label as f64);
    let fitted = SKPipeline::start::<f64>()
        .then(SKStandardScaler::build().build().unwrap())
        .finalize_regressor(
            sciencekit_linear_model::linear_regression::core_implementation::SKLinearRegression::build()
                .build()
                .unwrap(),
        )
        .unwrap()
        .fit(&features, targets.view())
        .unwrap();
    assert_eq!(fitted.predict(&features).unwrap().len(), 300);
}
