//! Tests for the export/import traits and the in-memory round-trip
//! (spec `model-export-contracts`).

use ndarray::array;

use super::mock_model::{MOCK_ESTIMATOR_NAME, SKMockLinearModel, mock_linear_model};
use super::{SKExportable, SKImportable, SKTensorDataType};

/// Export names the estimator, its scalar data_type, feature count and stable tensors.
#[test]
fn interop_traits_export_describes_the_model() {
    let model = mock_linear_model();

    let state = model.export_state().unwrap();

    assert_eq!(state.metadata().estimator_name(), MOCK_ESTIMATOR_NAME);
    assert_eq!(state.metadata().format_version(), 1);
    assert_eq!(state.metadata().scalar_data_type(), SKTensorDataType::F64);
    assert_eq!(state.metadata().feature_count(), 3);
    assert_eq!(state.number_of_tensors(), 2);
    assert_eq!(
        state
            .tensor("linear.coefficients")
            .map(|tensor| tensor.shape().to_vec()),
        Some(vec![3])
    );
    assert_eq!(
        state
            .tensor("linear.intercept")
            .map(|tensor| tensor.shape().to_vec()),
        Some(Vec::new())
    );
    assert_eq!(
        state
            .parameters()
            .get("solver")
            .and_then(super::SKParameterValue::as_string),
        Some("mock")
    );
}

/// Class labels survive export in index order.
#[test]
fn interop_traits_export_preserves_class_labels() {
    let model = mock_linear_model().with_class_labels(vec!["cat".to_string(), "dog".to_string()]);

    let state = model.export_state().unwrap();

    assert_eq!(
        state.metadata().class_labels(),
        Some(["cat".to_string(), "dog".to_string()].as_slice())
    );
}

/// A round-trip reproduces the same predictions within tolerance.
#[test]
fn interop_traits_round_trip_preserves_predictions() {
    let model = mock_linear_model();
    let state = model.export_state().unwrap();

    let restored = SKMockLinearModel::<f64>::import_state(&state).unwrap();

    for features in [
        array![1.0_f64, 2.0, 3.0],
        array![-1.0_f64, 0.0, 0.5],
        array![0.0_f64, 0.0, 0.0],
    ] {
        let expected = model.score(features.view());
        let found = restored.score(features.view());
        assert!(
            (expected - found).abs() < 1.0e-12,
            "prediction drifted: {expected} vs {found}"
        );
    }
    assert_eq!(restored, model);
}
