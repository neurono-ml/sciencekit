//! Tests for the fitted-model accessors (spec `typed-pipeline`).

use ndarray::{Array1, Array2};
use sciencekit_common::{SKBackendKind, SKBuilder, SKExecutionMode};
use sciencekit_impute::simple_imputer::SKSimpleImputer;
use sciencekit_linear_model::logistic_regression::SKLogisticRegression;
use sciencekit_preprocessing::standard_scaler::core_implementation::SKStandardScaler;

use crate::SKPipeline;

type FittedChain = (((), SKStandardScaler), SKSimpleImputer<f64>);
type FittedPipeline = crate::SKPipelineClassifier<
    f64,
    FittedChain,
    sciencekit_linear_model::logistic_regression::model::SKLogisticRegressionModel<f64>,
>;

fn fitted_pipeline() -> FittedPipeline {
    let features = Array2::from_shape_vec(
        (6, 2),
        vec![0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 5.0, 5.0, 6.0],
    )
    .unwrap();
    let labels = Array1::from_vec(vec![0_i64, 0, 0, 1, 1, 1]);
    SKPipeline::start::<f64>()
        .then(SKStandardScaler::build().build().unwrap())
        .then(SKSimpleImputer::build().build().unwrap())
        .finalize_classifier(SKLogisticRegression::build().build().unwrap())
        .unwrap()
        .fit(&features, labels.view())
        .unwrap()
}

#[test]
fn accessors_describe_the_composed_chain() {
    let fitted = fitted_pipeline();
    assert_eq!(fitted.number_of_stages(), 2);
    assert_eq!(fitted.number_of_features(), 2);
    assert_eq!(fitted.backend(), SKBackendKind::Faer);
    assert_eq!(fitted.execution_mode(), SKExecutionMode::Automatic);
}

#[test]
fn fitted_stages_expose_every_stage_model_in_order() {
    let fitted = fitted_pipeline();
    let ((_, scaler_model), imputer_model) = fitted.fitted_stages();
    assert_eq!(scaler_model.mean().len(), 2);
    assert_eq!(imputer_model.number_of_input_columns(), 2);
}

#[test]
fn the_final_model_is_accessible() {
    let fitted = fitted_pipeline();
    assert_eq!(fitted.final_model().number_of_classes(), 2);
}
