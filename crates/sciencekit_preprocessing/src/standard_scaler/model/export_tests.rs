//! Companion tests: the three scalers round-trip through the export contract.

use ndarray::Array2;
use sciencekit_common::{
    SKBackendKind, SKBuilder, SKExportable, SKFeatureTransformer, SKFloat, SKImportable,
    SKUnsupervisedFit, sk_assert_export_round_trip,
};

use super::SKStandardScalerModel;
use crate::minmax_scaler::core_implementation::SKMinMaxScaler;
use crate::robust_scaler::core_implementation::SKRobustScaler;
use crate::standard_scaler::core_implementation::SKStandardScaler;

fn features<F: SKFloat>() -> Array2<F> {
    Array2::from_shape_vec(
        (4, 3),
        vec![
            F::from(1.0).unwrap(),
            F::from(10.0).unwrap(),
            F::from(-2.0).unwrap(),
            F::from(2.0).unwrap(),
            F::from(20.0).unwrap(),
            F::from(0.0).unwrap(),
            F::from(3.0).unwrap(),
            F::from(30.0).unwrap(),
            F::from(2.0).unwrap(),
            F::from(4.0).unwrap(),
            F::from(40.0).unwrap(),
            F::from(4.0).unwrap(),
        ],
    )
    .unwrap()
}

#[test]
fn standard_scaler_round_trips_predictions_and_backend() {
    let matrix: Array2<f64> = features();
    let estimator = SKStandardScaler::build()
        .backend(SKBackendKind::MatrixMultiply)
        .build()
        .unwrap();
    let model = estimator.fit(&matrix).unwrap();
    let imported = sk_assert_export_round_trip::<f64, _>(&model).unwrap();
    assert_eq!(imported.backend(), SKBackendKind::MatrixMultiply);
    assert_eq!(
        model.transform(&matrix).unwrap(),
        imported.transform(&matrix).unwrap()
    );
}

#[test]
fn minmax_scaler_round_trips_predictions_and_range() {
    let matrix: Array2<f64> = features();
    let estimator = SKMinMaxScaler::<f64>::build()
        .feature_range((-1.0, 1.0))
        .clip(true)
        .backend(SKBackendKind::MatrixMultiply)
        .build()
        .unwrap();
    let model = estimator.fit(&matrix).unwrap();
    let imported = sk_assert_export_round_trip::<f64, _>(&model).unwrap();
    assert_eq!(imported.backend(), SKBackendKind::MatrixMultiply);
    assert_eq!(imported.feature_range(), (-1.0, 1.0));
    assert!(imported.clip());
    assert_eq!(
        model.transform(&matrix).unwrap(),
        imported.transform(&matrix).unwrap()
    );
}

#[test]
fn robust_scaler_round_trips_predictions_and_configuration() {
    let matrix: Array2<f64> = features();
    let estimator = SKRobustScaler::<f64>::build()
        .quantile_range((10.0, 90.0))
        .unit_variance(true)
        .backend(SKBackendKind::MatrixMultiply)
        .build()
        .unwrap();
    let model = estimator.fit(&matrix).unwrap();
    let imported = sk_assert_export_round_trip::<f64, _>(&model).unwrap();
    assert_eq!(imported.backend(), SKBackendKind::MatrixMultiply);
    assert_eq!(imported.quantile_range(), (10.0, 90.0));
    assert!(imported.unit_variance());
    assert_eq!(
        model.transform(&matrix).unwrap(),
        imported.transform(&matrix).unwrap()
    );
}

#[test]
fn imported_scaler_rejects_a_foreign_estimator_name() {
    let matrix: Array2<f64> = features();
    let model = SKStandardScaler::build()
        .build()
        .unwrap()
        .fit(&matrix)
        .unwrap();
    let mut state = model.export_state().unwrap();
    let metadata = state.metadata().clone();
    let foreign = sciencekit_common::SKModelMetadata::new(
        "SKForeignScaler",
        metadata.format_version(),
        metadata.scalar_data_type(),
        metadata.feature_count(),
    )
    .unwrap();
    state =
        sciencekit_common::SKModelState::new(foreign, state.tensors().values().cloned().collect())
            .unwrap();
    assert!(SKStandardScalerModel::<f64>::import_state(&state).is_err());
}

#[test]
fn scalar_round_trip_uses_f32_dtype() {
    let matrix: Array2<f32> = features();
    let model = SKStandardScaler::build()
        .build()
        .unwrap()
        .fit(&matrix)
        .unwrap();
    let state = model.export_state().unwrap();
    assert_eq!(
        state.metadata().scalar_data_type(),
        sciencekit_common::SKTensorDataType::F32
    );
}
