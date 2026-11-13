//! Every existing transformer composes through the stage adapter: the three
//! scalers, the simple imputer and the missing indicator.

use ndarray::{Array1, Array2};
use sciencekit_common::{SKBuilder, SKClassifierPredictor};
use sciencekit_impute::missing_indicator::SKMissingIndicator;
use sciencekit_impute::simple_imputer::SKSimpleImputer;
use sciencekit_linear_model::logistic_regression::SKLogisticRegression;
use sciencekit_preprocessing::minmax_scaler::core_implementation::SKMinMaxScaler;
use sciencekit_preprocessing::robust_scaler::core_implementation::SKRobustScaler;
use sciencekit_preprocessing::standard_scaler::core_implementation::SKStandardScaler;

use super::fixtures::{tiny_features, tiny_labels};
use crate::SKPipeline;

fn classifier() -> SKLogisticRegression<f64> {
    SKLogisticRegression::build().build().unwrap()
}

fn missing_features() -> Array2<f64> {
    Array2::from_shape_vec(
        (8, 2),
        vec![
            1.0,
            f64::NAN,
            2.0,
            3.0,
            f64::NAN,
            4.0,
            5.0,
            6.0,
            6.0,
            f64::NAN,
            7.0,
            8.0,
            f64::NAN,
            9.0,
            10.0,
            11.0,
        ],
    )
    .unwrap()
}

#[test]
fn the_standard_scaler_composes() {
    let fitted = SKPipeline::start::<f64>()
        .then(SKStandardScaler::build().build().unwrap())
        .finalize_classifier(classifier())
        .unwrap()
        .fit(&tiny_features(), tiny_labels().view())
        .unwrap();
    assert_eq!(fitted.predict_labels(&tiny_features()).unwrap().len(), 8);
}

#[test]
fn the_min_max_scaler_composes() {
    let fitted = SKPipeline::start::<f64>()
        .then(SKMinMaxScaler::build().build().unwrap())
        .finalize_classifier(classifier())
        .unwrap()
        .fit(&tiny_features(), tiny_labels().view())
        .unwrap();
    assert_eq!(fitted.predict_labels(&tiny_features()).unwrap().len(), 8);
}

#[test]
fn the_robust_scaler_composes() {
    let fitted = SKPipeline::start::<f64>()
        .then(SKRobustScaler::build().build().unwrap())
        .finalize_classifier(classifier())
        .unwrap()
        .fit(&tiny_features(), tiny_labels().view())
        .unwrap();
    assert_eq!(fitted.predict_labels(&tiny_features()).unwrap().len(), 8);
}

#[test]
fn the_simple_imputer_composes() {
    let fitted = SKPipeline::start::<f64>()
        .then(SKSimpleImputer::build().build().unwrap())
        .finalize_classifier(classifier())
        .unwrap()
        .fit(&tiny_features(), tiny_labels().view())
        .unwrap();
    assert_eq!(fitted.number_of_stages(), 1);
}

#[test]
fn the_missing_indicator_composes_with_the_imputer() {
    let labels = Array1::from_vec(vec![0, 0, 1, 1, 0, 1, 0, 1]);
    let fitted = SKPipeline::start::<f64>()
        .then(
            SKMissingIndicator::build()
                .error_on_new(false)
                .build()
                .unwrap(),
        )
        .then(SKSimpleImputer::build().build().unwrap())
        .finalize_classifier(classifier())
        .unwrap()
        .fit(&missing_features(), labels.view())
        .unwrap();
    assert_eq!(fitted.number_of_stages(), 2);
    assert_eq!(fitted.predict_labels(&missing_features()).unwrap().len(), 8);
}
