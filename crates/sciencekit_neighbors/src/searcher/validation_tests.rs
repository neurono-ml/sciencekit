//! Companion tests for the shared query validation helper.

use ndarray::array;
use sciencekit_common::SKError;

use super::sk_validate_query;

#[test]
fn validation_accepts_a_well_formed_query_batch() {
    let queries = array![[0.0_f64, 1.0], [2.0, 3.0]];
    assert!(
        sk_validate_query(queries.view(), 2, 10, 3).is_ok(),
        "a valid batch must pass"
    );
}

#[test]
fn validation_rejects_a_feature_count_mismatch_with_both_dimensions() {
    let queries = array![[0.0_f64, 1.0], [2.0, 3.0]];
    let error = sk_validate_query(queries.view(), 3, 10, 1).unwrap_err();

    match error {
        SKError::ShapeMismatch { expected, found } => {
            assert_eq!(expected, vec![2, 3]);
            assert_eq!(found, vec![2, 2]);
        }
        other => panic!("expected ShapeMismatch, found {other:?}"),
    }
}

#[test]
fn validation_rejects_a_non_finite_query_at_its_flat_position() {
    let queries = array![[0.0_f64, 1.0], [2.0, f64::NAN]];
    let error = sk_validate_query(queries.view(), 2, 10, 1).unwrap_err();

    match error {
        SKError::NonFiniteValue { position } => assert_eq!(position, 3),
        other => panic!("expected NonFiniteValue, found {other:?}"),
    }
}

#[test]
fn validation_rejects_zero_neighbors() {
    let queries = array![[0.0_f64, 1.0]];
    let error = sk_validate_query(queries.view(), 2, 10, 0).unwrap_err();

    match error {
        SKError::InvalidHyperparameter { name, .. } => assert_eq!(name, "neighbors_count"),
        other => panic!("expected InvalidHyperparameter, found {other:?}"),
    }
}

#[test]
fn validation_rejects_more_neighbors_than_training_rows() {
    let queries = array![[0.0_f64, 1.0]];
    let error = sk_validate_query(queries.view(), 2, 10, 11).unwrap_err();

    match error {
        SKError::InvalidHyperparameter { name, .. } => assert_eq!(name, "neighbors_count"),
        other => panic!("expected InvalidHyperparameter, found {other:?}"),
    }
}

#[test]
fn validation_accepts_an_empty_query_batch_with_matching_features() {
    let queries = ndarray::Array2::<f64>::zeros((0, 2));
    assert!(sk_validate_query(queries.view(), 2, 10, 3).is_ok());
}
