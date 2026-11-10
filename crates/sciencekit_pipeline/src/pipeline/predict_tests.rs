//! Tests for prediction delegation (spec `typed-pipeline`).

use ndarray::{Array1, Array2};
use sciencekit_common::{
    SKBuilder, SKClassifierPredictor, SKFeatureBlock, SKRegressorPredictor,
};
use sciencekit_impute::simple_imputer::SKSimpleImputer;
use sciencekit_linear_model::linear_regression::core_implementation::SKLinearRegression;
use sciencekit_linear_model::logistic_regression::SKLogisticRegression;
use sciencekit_preprocessing::standard_scaler::core_implementation::SKStandardScaler;

use crate::{SKConfiguredStages, SKPipeline};

fn training_features() -> Array2<f64> {
    Array2::from_shape_vec(
        (6, 2),
        vec![0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 5.0, 5.0, 6.0],
    )
    .unwrap()
}

fn new_features() -> Array2<f64> {
    Array2::from_shape_vec((2, 2), vec![0.5, 1.5, 4.5, 5.5]).unwrap()
}

fn new_block() -> SKFeatureBlock<f64> {
    SKFeatureBlock::from(new_features())
}

fn classifier_pipeline() -> crate::SKPipelineClassifier<
    f64,
    (((), SKStandardScaler), SKSimpleImputer<f64>),
    sciencekit_linear_model::logistic_regression::model::SKLogisticRegressionModel<f64>,
> {
    let labels = Array1::from_vec(vec![0_i64, 0, 0, 1, 1, 1]);
    SKPipeline::start::<f64>()
        .then(SKStandardScaler::build().build().unwrap())
        .then(SKSimpleImputer::build().build().unwrap())
        .finalize_classifier(SKLogisticRegression::build().build().unwrap())
        .unwrap()
        .fit(&training_features(), labels.view())
        .unwrap()
}

#[test]
fn classifier_labels_equal_transform_then_final_model() {
    let fitted = classifier_pipeline();
    let expected = fitted
        .final_model()
        .predict_labels(
            &fitted
                .stages()
                .transform_stages(
                    fitted.fitted_stages(),
                    &new_block(),
                )
                .unwrap(),
        )
        .unwrap();
    let actual = fitted.predict_labels(&new_features()).unwrap();
    assert_eq!(actual, expected);
}

#[test]
fn classifier_probabilities_equal_transform_then_final_model() {
    let fitted = classifier_pipeline();
    let expected = fitted
        .final_model()
        .predict_probabilities(
            &fitted
                .stages()
                .transform_stages(
                    fitted.fitted_stages(),
                    &new_block(),
                )
                .unwrap(),
        )
        .unwrap();
    let actual = fitted.predict_probabilities(&new_features()).unwrap();
    assert_eq!(actual, expected);
}

#[test]
fn regressor_responses_equal_transform_then_final_model() {
    let targets = Array1::from_vec(vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
    let fitted = SKPipeline::start::<f64>()
        .then(SKStandardScaler::build().build().unwrap())
        .finalize_regressor(SKLinearRegression::build().build().unwrap())
        .unwrap()
        .fit(&training_features(), targets.view())
        .unwrap();
    let expected = fitted
        .final_model()
        .predict(
            &fitted
                .stages()
                .transform_stages(
                    fitted.fitted_stages(),
                    &new_block(),
                )
                .unwrap(),
        )
        .unwrap();
    let actual = fitted.predict(&new_features()).unwrap();
    assert_eq!(actual, expected);
}
