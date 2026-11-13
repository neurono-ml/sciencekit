//! Round-trip tests for the exported linear-regression state
//! (spec `model-export-contracts`).

use ndarray::array;
use sciencekit_common::{
    SKBackendKind, SKBuilder, SKExportable, SKImportable, SKModelMetadata, SKModelState,
    SKParameterValue, SKRegressorPredictor, SKSupervisedFit,
};

use super::SKLinearRegressionModel;
use super::{SK_LINEAR_REGRESSION_ESTIMATOR_NAME, SK_LINEAR_REGRESSION_FORMAT_VERSION};
use crate::linear_regression::core_implementation::SKLinearRegression;

/// Fit a tiny exactly linear dataset with a known solution.
fn fitted_model() -> SKLinearRegressionModel<f64> {
    let features = array![[0.0_f64, 0.0], [1.0, 0.0], [0.0, 1.0], [2.0, 3.0]];
    let targets = array![1.0_f64, 3.0, 4.0, 14.0];
    let estimator = SKLinearRegression::build()
        .build()
        .expect("the default configuration is valid");
    estimator
        .fit(features.view(), targets.view())
        .expect("the synthetic dataset is exactly linear")
}

/// Exporting then importing a model preserves coefficients and predictions.
#[test]
fn export_import_round_trip_preserves_coefficients() {
    let model = fitted_model();
    let state = model.export_state().expect("export succeeds");
    assert_eq!(
        state.metadata().estimator_name(),
        SK_LINEAR_REGRESSION_ESTIMATOR_NAME
    );
    assert_eq!(
        state.metadata().format_version(),
        SK_LINEAR_REGRESSION_FORMAT_VERSION
    );
    let restored = SKLinearRegressionModel::<f64>::import_state(&state).expect("import succeeds");
    assert_eq!(restored.coefficients(), model.coefficients());
    assert_eq!(restored.intercept(), model.intercept());
}

/// A state carrying another estimator name is rejected before tensors are read.
#[test]
fn export_import_rejects_a_foreign_estimator() {
    let metadata =
        SKModelMetadata::new("SKOther", 1, sciencekit_common::SKTensorDataType::F64, 2).unwrap();
    let state = SKModelState::new(metadata, Vec::new()).expect("an empty state is valid");
    let error = SKLinearRegressionModel::<f64>::import_state(&state)
        .expect_err("a foreign estimator must be rejected");
    assert!(matches!(
        error,
        sciencekit_common::SKError::UnsupportedRepresentation { .. }
    ));
}

/// Incomplete states are rejected instead of producing a half-built model.
#[test]
fn export_import_rejects_missing_tensors() {
    let metadata = SKModelMetadata::new(
        SK_LINEAR_REGRESSION_ESTIMATOR_NAME,
        SK_LINEAR_REGRESSION_FORMAT_VERSION,
        sciencekit_common::SKTensorDataType::F64,
        2,
    )
    .unwrap();
    let state = SKModelState::new(metadata, Vec::new()).expect("an empty state is valid");
    assert!(SKLinearRegressionModel::<f64>::import_state(&state).is_err());
}

/// Exporting then importing a model preserves every host and device backend kind.
#[test]
fn export_import_round_trip_preserves_device_and_host_backends() {
    let features = array![[0.0_f64, 0.0], [1.0, 0.0], [0.0, 1.0], [2.0, 3.0]];
    let targets = array![1.0_f64, 3.0, 4.0, 14.0];

    for kind in SKBackendKind::ALL {
        let estimator = SKLinearRegression::build()
            .backend(kind)
            .build()
            .expect("every backend kind builds a valid estimator");
        let model = estimator
            .fit(features.view(), targets.view())
            .expect("fit succeeds");
        let state = model.export_state().expect("export succeeds");
        let restored =
            SKLinearRegressionModel::<f64>::import_state(&state).expect("import succeeds");
        assert_eq!(
            restored.backend(),
            kind,
            "backend `{kind:?}` must survive export/import"
        );
    }
}

/// Importing also accepts the lowercase `SKBackendKind::name()` spellings.
#[test]
fn import_accepts_lowercase_backend_names() {
    let base_state = fitted_model().export_state().expect("export succeeds");
    for kind in [
        SKBackendKind::OpenCl,
        SKBackendKind::Cuda,
        SKBackendKind::Rocm,
    ] {
        let metadata = SKModelMetadata::new(
            SK_LINEAR_REGRESSION_ESTIMATOR_NAME,
            SK_LINEAR_REGRESSION_FORMAT_VERSION,
            sciencekit_common::SKTensorDataType::F64,
            base_state.metadata().feature_count(),
        )
        .unwrap()
        .with_parameter("backend", SKParameterValue::from(kind.name().to_string()))
        .unwrap();
        let tensors = base_state.tensors().values().cloned().collect();
        let state = SKModelState::new(metadata, tensors).unwrap();
        let restored =
            SKLinearRegressionModel::<f64>::import_state(&state).expect("import succeeds");
        assert_eq!(restored.backend(), kind);
    }
}

#[test]
fn linear_regression_import_rejects_a_future_format_version() {
    let features = array![[0.0_f64], [1.0]];
    let targets = array![0.0_f64, 2.0];
    let model = SKLinearRegression::build()
        .build()
        .unwrap()
        .fit(features.view(), targets.view())
        .unwrap();
    let state = SKExportable::<f64>::export_state(&model).unwrap();
    let original = state.metadata();
    let future = sciencekit_common::SKModelMetadata::new(
        original.estimator_name(),
        original.format_version() + 1,
        original.scalar_data_type(),
        original.feature_count(),
    )
    .unwrap();
    let broken =
        sciencekit_common::SKModelState::new(future, state.tensors().values().cloned().collect())
            .unwrap();
    assert!(matches!(
        SKLinearRegressionModel::<f64>::import_state(&broken),
        Err(sciencekit_common::SKError::UnsupportedRepresentation { .. })
    ));
}

#[test]
fn linear_regression_import_ignores_an_unexpected_extra_tensor() {
    let features = array![[0.0_f64], [1.0], [2.0]];
    let targets = array![1.0_f64, 3.0, 5.0];
    let model = SKLinearRegression::build()
        .build()
        .unwrap()
        .fit(features.view(), targets.view())
        .unwrap();
    let state = SKExportable::<f64>::export_state(&model).unwrap();
    let mut tensors: Vec<_> = state.tensors().values().cloned().collect();
    tensors.push(
        sciencekit_common::SKTensor::floating("linear.unexpected", vec![1], vec![42.0]).unwrap(),
    );
    let extended = sciencekit_common::SKModelState::new(state.metadata().clone(), tensors).unwrap();
    let imported = SKLinearRegressionModel::<f64>::import_state(&extended).unwrap();
    assert_eq!(
        imported.predict(features.view()).unwrap(),
        model.predict(features.view()).unwrap()
    );
}
