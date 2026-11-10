//! Determinism, thread-safety and concurrent prediction tests (spec
//! `typed-pipeline`).

use ndarray::{Array1, Array2};
use sciencekit_common::{
    SKBuilder, SKClassifierPredictor, SKRegressorPredictor,
};
use sciencekit_impute::simple_imputer::SKSimpleImputer;
use sciencekit_linear_model::linear_regression::core_implementation::SKLinearRegression;
use sciencekit_linear_model::logistic_regression::SKLogisticRegression;
use sciencekit_preprocessing::standard_scaler::core_implementation::SKStandardScaler;

use crate::SKPipeline;

type ClassifierPipeline = crate::SKPipelineClassifier<
    f64,
    (((), SKStandardScaler), SKSimpleImputer<f64>),
    sciencekit_linear_model::logistic_regression::model::SKLogisticRegressionModel<f64>,
>;

fn features() -> Array2<f64> {
    Array2::from_shape_vec(
        (8, 2),
        vec![
            0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 5.0, 5.0, 6.0, 6.0, 7.0, 7.0, 8.0,
        ],
    )
    .unwrap()
}

fn labels() -> Array1<i64> {
    Array1::from_vec(vec![0, 0, 0, 0, 1, 1, 1, 1])
}

fn new_features() -> Array2<f64> {
    Array2::from_shape_vec((4, 2), vec![0.5, 1.5, 2.5, 3.5, 5.5, 6.5, 7.0, 8.0]).unwrap()
}

fn build() -> ClassifierPipeline {
    SKPipeline::start::<f64>()
        .then(SKStandardScaler::build().build().unwrap())
        .then(SKSimpleImputer::build().build().unwrap())
        .finalize_classifier(SKLogisticRegression::build().build().unwrap())
        .unwrap()
        .fit(&features(), labels().view())
        .unwrap()
}

#[test]
fn the_same_pipeline_fitted_twice_produces_identical_predictions() {
    let first = build();
    let second = build();
    assert_eq!(
        first.predict_labels(&new_features()).unwrap(),
        second.predict_labels(&new_features()).unwrap()
    );
    assert_eq!(
        first.predict_probabilities(&new_features()).unwrap(),
        second.predict_probabilities(&new_features()).unwrap()
    );
}

#[test]
fn the_fitted_pipeline_is_send_and_sync() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<ClassifierPipeline>();
    let targets = Array1::from_vec(vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]);
    let regressor = SKPipeline::start::<f64>()
        .then(SKStandardScaler::build().build().unwrap())
        .finalize_regressor(SKLinearRegression::build().build().unwrap())
        .unwrap()
        .fit(&features(), targets.view())
        .unwrap();
    assert_send_sync::<crate::SKPipelineRegressor<
        f64,
        ((), SKStandardScaler),
        sciencekit_linear_model::linear_regression::model::SKLinearRegressionModel<f64>,
    >>();
    assert_eq!(regressor.predict(&new_features()).unwrap().len(), 4);
}

#[test]
fn concurrent_prediction_agrees_with_the_sequential_result() {
    let fitted = build();
    let expected = fitted.predict_labels(&new_features()).unwrap();
    std::thread::scope(|scope| {
        let mut handles = Vec::new();
        for _ in 0..8 {
            let pipeline = &fitted;
            handles.push(scope.spawn(move || pipeline.predict_labels(&new_features()).unwrap()));
        }
        for handle in handles {
            assert_eq!(handle.join().unwrap(), expected);
        }
    });
}
