//! Companion tests for the scope learned at fit (spec `missing-indicator`).

use ndarray::{Array2, array};
use sciencekit_common::{SKBuilder, SKError, SKUnsupervisedFit};

use super::core_implementation::SKMissingIndicator;
use super::scope::SKMissingFeatureScope;
use super::test_support::csr;

#[test]
fn missing_only_scope_is_exactly_the_missing_columns() {
    let model = SKMissingIndicator::<f64>::build()
        .build()
        .unwrap()
        .fit(&array![
            [1.0, 2.0, 3.0],
            [4.0, f64::NAN, 6.0],
            [f64::NAN, 8.0, 9.0]
        ])
        .unwrap();
    assert_eq!(model.scope(), &[0, 1]);
    assert_eq!(model.number_of_input_columns(), 3);
    assert_eq!(model.number_of_output_columns(), 2);
}

#[test]
fn all_scope_covers_every_feature() {
    let model = SKMissingIndicator::<f64>::build()
        .features(SKMissingFeatureScope::All)
        .build()
        .unwrap()
        .fit(&array![[1.0, f64::NAN], [3.0, 4.0]])
        .unwrap();
    assert_eq!(model.scope(), &[0, 1]);
    assert_eq!(model.number_of_output_columns(), 2);
}

#[test]
fn the_scope_is_deterministic_across_fits() {
    let features = array![[1.0, f64::NAN, 3.0], [4.0, 5.0, f64::NAN]];
    let estimator = SKMissingIndicator::<f64>::build().build().unwrap();
    let first = estimator.fit(&features).unwrap();
    let second = estimator.fit(&features).unwrap();
    assert_eq!(first.scope(), second.scope());
    assert_eq!(
        first.number_of_input_columns(),
        second.number_of_input_columns()
    );
}

#[test]
fn sparse_scope_counts_the_explicit_missing_entries() {
    let features = csr((2, 2), &[(0, 0, 1.0), (0, 1, f64::NAN), (1, 0, 3.0)]);
    let model = SKMissingIndicator::<f64>::build()
        .build()
        .unwrap()
        .fit(features.view())
        .unwrap();
    assert_eq!(model.scope(), &[1], "implicit zeros are observed under NaN");
}

#[test]
fn sparse_zero_marker_claims_the_implicit_zeros() {
    let features = csr((2, 2), &[(0, 0, 1.0)]);
    let model = SKMissingIndicator::<f64>::build()
        .missing_values(0.0)
        .build()
        .unwrap()
        .fit(features.view())
        .unwrap();
    assert_eq!(model.scope(), &[0, 1]);
}

#[test]
fn empty_input_is_rejected_at_fit() {
    let error = SKMissingIndicator::<f64>::build()
        .build()
        .unwrap()
        .fit(&Array2::<f64>::zeros((0, 2)))
        .map(|_| ())
        .expect_err("an empty matrix must fail");
    assert!(matches!(error, SKError::ShapeMismatch { .. }));
}

#[test]
fn the_model_carries_the_configured_backend() {
    use sciencekit_common::SKBackendKind;

    let model = SKMissingIndicator::<f64>::build()
        .backend(SKBackendKind::MatrixMultiply)
        .build()
        .unwrap()
        .fit(&array![[1.0], [f64::NAN]])
        .unwrap();
    assert_eq!(model.backend(), SKBackendKind::MatrixMultiply);
}
