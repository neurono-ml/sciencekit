//! Companion test: a fitted pipeline nests every stage and the final model.

use ndarray::{Array1, Array2};
use sciencekit_common::{SKBuilder, SKExportable, SKFloat, SKImportable};
use sciencekit_linear_model::logistic_regression::SKLogisticRegression;
use sciencekit_preprocessing::standard_scaler::core_implementation::SKStandardScaler;

use crate::SKPipeline;

#[test]
fn pipeline_export_nests_stages_and_the_final_model() {
    let features = Array2::from_shape_vec((3, 2), vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let labels = Array1::from_vec(vec![0_i64, 1, 0]);
    let pipeline = SKPipeline::start::<f64>()
        .then(SKStandardScaler::build().build().unwrap())
        .finalize_classifier(SKLogisticRegression::build().build().unwrap())
        .unwrap();
    let fitted = pipeline.fit(&features, labels.view()).unwrap();
    let state = fitted.export_state().unwrap();
    assert!(
        state
            .tensor("pipeline.stage.0.tensor.standard_scaler.mean")
            .is_some()
    );
    assert!(
        state
            .tensor("pipeline.final.tensor.logistic.coefficients")
            .is_some()
    );
    assert_eq!(
        state
            .metadata()
            .parameter("number_of_stages")
            .and_then(sciencekit_common::SKParameterValue::as_integer),
        Some(1)
    );
}

fn assert_pipeline_import_is_explicitly_unsupported<F, Stages, FinalModel>(
    model: &crate::pipeline::SKPipelineClassifier<F, Stages, FinalModel>,
) where
    F: SKFloat,
    Stages: crate::stages::SKConfiguredStages<F>,
    Stages::FittedStages: crate::export::SKExportableStages<F>,
    FinalModel: SKExportable<F>,
{
    let state = model.export_state().unwrap();
    assert!(matches!(
        crate::pipeline::SKPipelineClassifier::<F, Stages, FinalModel>::import_state(&state),
        Err(sciencekit_common::SKError::UnsupportedOperation {
            operation: "import_state",
            ..
        })
    ));
}

#[test]
fn typed_pipeline_import_returns_a_typed_unsupported_operation() {
    let features = Array2::from_shape_vec((3, 2), vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let labels = Array1::from_vec(vec![0_i64, 1, 0]);
    let pipeline = SKPipeline::start::<f64>()
        .then(SKStandardScaler::build().build().unwrap())
        .finalize_classifier(SKLogisticRegression::build().build().unwrap())
        .unwrap();
    let fitted = pipeline.fit(&features, labels.view()).unwrap();
    assert_pipeline_import_is_explicitly_unsupported(&fitted);
}
