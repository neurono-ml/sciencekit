//! Companion tests: the L1-regularized models round-trip their solutions.

use ndarray::array;
use sciencekit_common::{
    SKBackendKind, SKBuilder, SKImportable, SKRegressorPredictor, SKSupervisedFit,
    sk_assert_export_round_trip,
};

use crate::coordinate_descent::SKSelection;
use crate::elastic_net::core_implementation::SKElasticNet;
use crate::lasso::core_implementation::SKLasso;
use crate::lasso::model::SKLassoModel;

#[test]
fn lasso_round_trips_multi_alpha_solutions() {
    let features = array![[0.0_f64], [1.0], [2.0], [3.0]];
    let targets = array![1.0_f64, 3.0, 5.0, 7.0];
    let model = SKLasso::<f64>::build()
        .alpha(vec![0.01_f64, 0.1])
        .selection(SKSelection::Random)
        .backend(SKBackendKind::MatrixMultiply)
        .build()
        .unwrap()
        .fit(features.view(), targets.view())
        .unwrap();
    let imported = sk_assert_export_round_trip::<f64, _>(&model).unwrap();
    assert_eq!(imported.alphas(), model.alphas());
    assert_eq!(imported.coefficients(), model.coefficients());
    assert_eq!(imported.backend(), SKBackendKind::MatrixMultiply);
    let probe = array![[0.5_f64], [2.5]];
    assert_eq!(
        model.predict_with_alpha_index(1, probe.view()).unwrap(),
        imported.predict_with_alpha_index(1, probe.view()).unwrap()
    );
}

#[test]
fn elastic_net_round_trips_the_mixing_ratio() {
    let features = array![[0.0_f64, 1.0], [1.0, 0.0], [2.0, 2.0], [3.0, 1.0]];
    let targets = array![1.0_f64, 2.0, 5.0, 6.0];
    let model = SKElasticNet::<f64>::build()
        .alpha(0.05)
        .l1_ratio(0.7)
        .backend(SKBackendKind::MatrixMultiply)
        .build()
        .unwrap()
        .fit(features.view(), targets.view())
        .unwrap();
    let imported = sk_assert_export_round_trip::<f64, _>(&model).unwrap();
    assert_eq!(imported.l1_ratio(), model.l1_ratio());
    assert_eq!(imported.backend(), SKBackendKind::MatrixMultiply);
    assert_eq!(
        imported.predict(features.view()).unwrap(),
        model.predict(features.view()).unwrap()
    );
}

#[test]
fn lasso_import_rejects_a_shape_mismatch() {
    let features = array![[0.0_f64], [1.0]];
    let targets = array![0.0_f64, 2.0];
    let model = SKLasso::<f64>::build()
        .build()
        .unwrap()
        .fit(features.view(), targets.view())
        .unwrap();
    let state = sciencekit_common::SKExportable::<f64>::export_state(&model).unwrap();
    let mut tensors: Vec<_> = state.tensors().values().cloned().collect();
    tensors.retain(|tensor| tensor.name() != "lasso.coefficients");
    tensors.push(
        sciencekit_common::SKTensor::floating("lasso.coefficients", vec![1, 2], vec![0.0; 2])
            .unwrap(),
    );
    let broken = sciencekit_common::SKModelState::new(state.metadata().clone(), tensors).unwrap();
    assert!(SKLassoModel::<f64>::import_state(&broken).is_err());
}
