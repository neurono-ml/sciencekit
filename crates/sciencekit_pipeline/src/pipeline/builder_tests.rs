//! Tests for the typed builder and the finalizers (spec `typed-pipeline`).

use ndarray::{Array1, Array2};
use sciencekit_common::{SKBuilder, SKError};
use sciencekit_impute::simple_imputer::SKSimpleImputer;
use sciencekit_linear_model::linear_regression::core_implementation::SKLinearRegression;
use sciencekit_linear_model::logistic_regression::SKLogisticRegression;
use sciencekit_preprocessing::standard_scaler::core_implementation::SKStandardScaler;

use crate::SKPipeline;

fn features() -> Array2<f64> {
    Array2::from_shape_vec(
        (6, 2),
        vec![0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 5.0, 5.0, 6.0],
    )
    .unwrap()
}

fn labels() -> Array1<i64> {
    Array1::from_vec(vec![0, 0, 0, 1, 1, 1])
}

fn scaler() -> SKStandardScaler {
    SKStandardScaler::build().build().unwrap()
}

fn imputer() -> SKSimpleImputer<f64> {
    SKSimpleImputer::build().build().unwrap()
}

fn classifier() -> SKLogisticRegression<f64> {
    SKLogisticRegression::build().build().unwrap()
}

fn regressor() -> SKLinearRegression {
    SKLinearRegression::build().build().unwrap()
}

#[test]
fn two_transformers_and_a_classifier_compile_and_fit() {
    let pipeline = SKPipeline::start::<f64>()
        .then(scaler())
        .then(imputer())
        .finalize_classifier(classifier())
        .unwrap();
    let fitted = pipeline.fit(&features(), labels().view()).unwrap();
    assert_eq!(fitted.number_of_stages(), 2);
    assert_eq!(fitted.number_of_features(), 2);
}

#[test]
fn a_regressor_finalizer_compiles_and_fits() {
    let targets = Array1::from_vec(vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
    let pipeline = SKPipeline::start::<f64>()
        .then(scaler())
        .finalize_regressor(regressor())
        .unwrap();
    let fitted = pipeline.fit(&features(), targets.view()).unwrap();
    assert_eq!(fitted.number_of_stages(), 1);
}

#[test]
fn an_empty_pipeline_is_rejected_at_finalization() {
    let error = SKPipeline::start::<f64>()
        .finalize_classifier(classifier())
        .err()
        .expect("an empty pipeline must be rejected");
    assert!(matches!(error, SKError::InvalidHyperparameter { .. }));
}

#[test]
fn an_empty_regressor_pipeline_is_rejected_at_finalization() {
    let error = SKPipeline::start::<f64>()
        .finalize_regressor(regressor())
        .err()
        .expect("an empty pipeline must be rejected");
    assert!(matches!(error, SKError::InvalidHyperparameter { .. }));
}
