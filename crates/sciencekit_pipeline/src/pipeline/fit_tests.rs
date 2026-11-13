//! Tests for the pipeline fit walk (spec `typed-pipeline`).

use ndarray::{Array1, Array2};
use sciencekit_common::{SKBuilder, SKClassifierPredictor};
use sciencekit_impute::simple_imputer::SKSimpleImputer;
use sciencekit_linear_model::logistic_regression::SKLogisticRegression;
use sciencekit_preprocessing::standard_scaler::core_implementation::SKStandardScaler;

use crate::SKPipeline;

fn raw_features() -> Array2<f64> {
    Array2::from_shape_vec((3, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap()
}

fn labels() -> Array1<i64> {
    Array1::from_vec(vec![0, 1, 0])
}

#[test]
fn a_scaler_imputer_classifier_chain_fits_each_stage_on_the_previous_output() {
    let pipeline = SKPipeline::start::<f64>()
        .then(SKStandardScaler::build().build().unwrap())
        .then(SKSimpleImputer::build().build().unwrap())
        .finalize_classifier(SKLogisticRegression::build().build().unwrap())
        .unwrap();
    let fitted = pipeline.fit(&raw_features(), labels().view()).unwrap();

    let ((_, scaler_model), imputer_model) = fitted.fitted_stages();
    // The scaler learns the raw column means [3, 4].
    assert!((scaler_model.mean()[0] - 3.0).abs() < 1e-12);
    assert!((scaler_model.mean()[1] - 4.0).abs() < 1e-12);
    // The imputer is fitted on the *scaled* block, whose column means are zero,
    // not on the raw data (whose means are [3, 4]).
    assert!(imputer_model.statistics()[0].abs() < 1e-12);
    assert!(imputer_model.statistics()[1].abs() < 1e-12);

    // The final estimator consumed two features over the three training rows.
    assert_eq!(fitted.final_model().feature_count(), 2);
    assert_eq!(fitted.final_model().row_count(), 3);
    assert_eq!(fitted.number_of_features(), 2);
}

#[test]
fn the_same_pipeline_can_be_refitted_without_touching_its_configuration() {
    let features = raw_features();
    let targets = labels();
    let build = || {
        SKPipeline::start::<f64>()
            .then(SKStandardScaler::build().build().unwrap())
            .then(SKSimpleImputer::build().build().unwrap())
            .finalize_classifier(SKLogisticRegression::build().build().unwrap())
            .unwrap()
    };
    let first = build().fit(&features, targets.view()).unwrap();
    let second = build().fit(&features, targets.view()).unwrap();
    assert_eq!(first.number_of_features(), second.number_of_features());
    assert_eq!(first.number_of_stages(), second.number_of_stages());
    let first_labels = first.predict_labels(features.view()).unwrap();
    let second_labels = second.predict_labels(features.view()).unwrap();
    assert_eq!(first_labels, second_labels);
}
