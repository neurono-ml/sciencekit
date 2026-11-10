//! Small-data acceptance: classifier, regressor, nested pipelines, dense and
//! sparse input, and metric production.

use sciencekit_common::{SKBuilder, SKClassifierPredictor, SKRegressorPredictor};
use sciencekit_impute::simple_imputer::SKSimpleImputer;
use sciencekit_linear_model::linear_regression::core_implementation::SKLinearRegression;
use sciencekit_linear_model::logistic_regression::SKLogisticRegression;
use sciencekit_metrics::classification::sk_accuracy_score;
use sciencekit_preprocessing::standard_scaler::core_implementation::SKStandardScaler;

use super::fixtures::{
    tiny_features, tiny_labels, tiny_regression_targets, tiny_sparse_features,
};
use crate::SKPipeline;

fn classifier_pipeline() -> crate::SKPipelineClassifier<
    f64,
    (((), SKStandardScaler), SKSimpleImputer<f64>),
    sciencekit_linear_model::logistic_regression::model::SKLogisticRegressionModel<f64>,
> {
    SKPipeline::start::<f64>()
        .then(SKStandardScaler::build().build().unwrap())
        .then(SKSimpleImputer::build().build().unwrap())
        .finalize_classifier(SKLogisticRegression::build().build().unwrap())
        .unwrap()
        .fit(&tiny_features(), tiny_labels().view())
        .unwrap()
}

#[test]
fn a_tiny_classifier_pipeline_produces_metrics() {
    let fitted = classifier_pipeline();
    let predictions = fitted.predict_labels(&tiny_features()).unwrap();
    let truth: Vec<i64> = tiny_labels().to_vec();
    let predicted: Vec<i64> = predictions.to_vec();
    let accuracy: f64 = sk_accuracy_score(&truth, &predicted).unwrap();
    assert!(accuracy >= 0.75, "accuracy was {accuracy}");
}

#[test]
fn a_tiny_regressor_pipeline_predicts_every_row() {
    let fitted = SKPipeline::start::<f64>()
        .then(SKStandardScaler::build().build().unwrap())
        .finalize_regressor(SKLinearRegression::build().build().unwrap())
        .unwrap()
        .fit(&tiny_features(), tiny_regression_targets().view())
        .unwrap();
    let responses = fitted.predict(&tiny_features()).unwrap();
    assert_eq!(responses.len(), tiny_features().nrows());
    assert!(responses.iter().all(|value| value.is_finite()));
}

#[test]
fn a_pipeline_composed_from_an_inner_builder_nests_its_stage_tuple() {
    // The inner builder already records one stage; extending it from the outer
    // scope produces the nested tuple shape the fitted models mirror.
    let inner = SKPipeline::start::<f64>().then(SKStandardScaler::build().build().unwrap());
    let fitted = inner
        .then(SKSimpleImputer::build().build().unwrap())
        .finalize_classifier(SKLogisticRegression::build().build().unwrap())
        .unwrap()
        .fit(&tiny_features(), tiny_labels().view())
        .unwrap();
    assert_eq!(fitted.number_of_stages(), 2);
    let ((_, _), _) = fitted.fitted_stages();
    assert_eq!(fitted.predict_labels(&tiny_features()).unwrap().len(), 8);
}

#[test]
fn a_sparse_classifier_pipeline_matches_the_dense_one() {
    let dense = classifier_pipeline();
    let sparse = SKPipeline::start::<f64>()
        .then(SKSimpleImputer::build().build().unwrap())
        .finalize_classifier(SKLogisticRegression::build().build().unwrap())
        .unwrap()
        .fit(tiny_sparse_features().view(), tiny_labels().view())
        .unwrap();
    let dense_predictions = dense.predict_labels(&tiny_features()).unwrap();
    let sparse_predictions = sparse
        .predict_labels(tiny_sparse_features().view())
        .unwrap();
    assert_eq!(dense_predictions.len(), sparse_predictions.len());
    assert_eq!(dense_predictions, sparse_predictions);
}

#[test]
fn classifier_probabilities_are_a_distribution() {
    let fitted = classifier_pipeline();
    let probabilities = fitted.predict_probabilities(&tiny_features()).unwrap();
    for row in probabilities.rows() {
        let total: f64 = row.iter().sum();
        assert!((total - 1.0).abs() < 1e-9);
    }
}
