//! Companion tests: linear regression and ridge round-trip their state.

use ndarray::array;
use sciencekit_common::{
    SKBackendKind, SKBuilder, SKExportable, SKImportable, SKRegressorPredictor, SKSupervisedFit,
    sk_assert_export_round_trip,
};

use crate::linear_regression::core_implementation::SKLinearRegression;
use crate::linear_regression::model::SKLinearRegressionModel;
use crate::ridge::SKRidge;
use crate::ridge::builder::SKRidgeSolver;
use crate::ridge::model::SKRidgeModel;

#[test]
fn linear_regression_round_trips_coefficients_rank_and_backend() {
    let features = array![[0.0_f64, 0.0], [1.0, 0.0], [0.0, 1.0], [2.0, 3.0]];
    let targets = array![1.0_f64, 3.0, 4.0, 14.0];
    let model = SKLinearRegression::build()
        .backend(SKBackendKind::MatrixMultiply)
        .build()
        .unwrap()
        .fit(features.view(), targets.view())
        .unwrap();
    let imported = sk_assert_export_round_trip::<f64, _>(&model).unwrap();
    assert_eq!(imported.coefficients(), model.coefficients());
    assert_eq!(imported.intercept(), model.intercept());
    assert_eq!(imported.rank(), model.rank());
    assert_eq!(imported.backend(), SKBackendKind::MatrixMultiply);
    let probe = array![[1.0, 2.0], [3.0, 4.0]];
    assert_eq!(
        model.predict(probe.view()).unwrap(),
        imported.predict(probe.view()).unwrap()
    );
}

#[test]
fn ridge_round_trips_multi_alpha_solutions() {
    let features = array![[0.0_f64], [1.0], [2.0], [3.0]];
    let targets = array![1.0_f64, 3.0, 5.0, 7.0];
    let model = SKRidge::<f64>::build()
        .alpha(vec![0.5_f64, 1.5])
        .solver(SKRidgeSolver::Svd)
        .backend(SKBackendKind::MatrixMultiply)
        .build()
        .unwrap()
        .fit(features.view(), targets.view())
        .unwrap();
    let imported = sk_assert_export_round_trip::<f64, _>(&model).unwrap();
    assert_eq!(imported.alphas(), model.alphas());
    assert_eq!(imported.coefficients(), model.coefficients());
    assert_eq!(imported.intercepts(), model.intercepts());
    assert_eq!(imported.backend(), SKBackendKind::MatrixMultiply);
    let probe = array![[0.5_f64], [2.5]];
    assert_eq!(
        model.predict_with_alpha_index(1, probe.view()).unwrap(),
        imported.predict_with_alpha_index(1, probe.view()).unwrap()
    );
}

#[test]
fn linear_regression_rejects_a_missing_tensor() {
    let features = array![[0.0_f64], [1.0]];
    let targets = array![0.0_f64, 2.0];
    let model = SKLinearRegression::build()
        .build()
        .unwrap()
        .fit(features.view(), targets.view())
        .unwrap();
    let state = SKExportable::<f64>::export_state(&model).unwrap();
    let tensors: Vec<_> = state
        .tensors()
        .iter()
        .filter(|(name, _)| name.as_str() != "linear.coefficients")
        .map(|(_, tensor)| tensor.clone())
        .collect();
    let broken = sciencekit_common::SKModelState::new(state.metadata().clone(), tensors).unwrap();
    assert!(SKLinearRegressionModel::<f64>::import_state(&broken).is_err());
}

#[test]
fn ridge_import_rejects_a_coefficient_shape_mismatch() {
    let features = array![[0.0_f64], [1.0]];
    let targets = array![0.0_f64, 2.0];
    let model = SKRidge::<f64>::build()
        .build()
        .unwrap()
        .fit(features.view(), targets.view())
        .unwrap();
    let state = SKExportable::<f64>::export_state(&model).unwrap();
    let mut tensors: Vec<_> = state.tensors().values().cloned().collect();
    tensors.retain(|tensor| tensor.name() != "ridge.coefficients");
    tensors.push(
        sciencekit_common::SKTensor::floating("ridge.coefficients", vec![1, 2], vec![0.0; 2])
            .unwrap(),
    );
    let broken = sciencekit_common::SKModelState::new(state.metadata().clone(), tensors).unwrap();
    assert!(SKRidgeModel::<f64>::import_state(&broken).is_err());
}
