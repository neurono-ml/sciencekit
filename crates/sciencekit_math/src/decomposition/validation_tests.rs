//! Companion tests for the validation of the randomized range finder (spec
//! `randomized-svd`).

use ndarray::{Array1, Array2};
use sciencekit_common::{SKDataView, SKError};

use super::randomized::sk_randomized_svd;

/// Asserts that the error is the expected variant and returns it.
fn assert_invalid_hyperparameter(result: Result<(), SKError>, name: &str) {
    match result {
        Err(SKError::InvalidHyperparameter { name: observed, .. }) => {
            assert_eq!(observed, name, "the wrong hyperparameter was reported");
        }
        Err(other) => panic!("expected an invalid hyperparameter, got {other:?}"),
        Ok(()) => panic!("the invalid configuration was accepted"),
    }
}

#[test]
fn randomized_validation_rejects_an_empty_matrix() {
    let empty = Array2::<f64>::zeros((0, 4));
    match sk_randomized_svd(SKDataView::Dense(empty.view()), 1, 1, 2, 0, None) {
        Err(SKError::EmptyInput { operation }) => assert_eq!(operation, "randomized_svd"),
        Err(other) => panic!("expected an empty input error, got {other:?}"),
        Ok(_) => panic!("the empty matrix was accepted"),
    }
}

#[test]
fn randomized_validation_rejects_a_matrix_without_columns() {
    let empty = Array2::<f64>::zeros((4, 0));
    match sk_randomized_svd(SKDataView::Dense(empty.view()), 1, 1, 2, 0, None) {
        Err(SKError::EmptyInput { .. }) => {}
        Err(other) => panic!("expected an empty input error, got {other:?}"),
        Ok(_) => panic!("the empty matrix was accepted"),
    }
}

#[test]
fn randomized_validation_rejects_zero_components() {
    let matrix = Array2::<f64>::zeros((6, 4));
    let result = sk_randomized_svd(SKDataView::Dense(matrix.view()), 0, 1, 2, 0, None);
    assert_invalid_hyperparameter(result.map(|_| ()), "number_of_components");
}

#[test]
fn randomized_validation_rejects_excessive_components() {
    let matrix = Array2::<f64>::zeros((6, 4));
    let result = sk_randomized_svd(SKDataView::Dense(matrix.view()), 5, 1, 2, 0, None);
    assert_invalid_hyperparameter(result.map(|_| ()), "number_of_components");
}

#[test]
fn randomized_validation_rejects_zero_power_iterations() {
    let matrix = Array2::<f64>::zeros((6, 4));
    let result = sk_randomized_svd(SKDataView::Dense(matrix.view()), 2, 0, 2, 0, None);
    assert_invalid_hyperparameter(result.map(|_| ()), "number_of_power_iterations");
}

#[test]
fn randomized_validation_rejects_zero_oversampling() {
    let matrix = Array2::<f64>::zeros((6, 4));
    let result = sk_randomized_svd(SKDataView::Dense(matrix.view()), 2, 1, 0, 0, None);
    assert_invalid_hyperparameter(result.map(|_| ()), "oversampling");
}

#[test]
fn randomized_validation_rejects_a_mismatched_centering_vector() {
    let matrix = Array2::<f64>::zeros((6, 4));
    let mean = Array1::<f64>::zeros(3);
    match sk_randomized_svd(
        SKDataView::Dense(matrix.view()),
        2,
        1,
        2,
        0,
        Some(mean.view()),
    ) {
        Err(SKError::ShapeMismatch { expected, found }) => {
            assert_eq!(expected, vec![4, 1]);
            assert_eq!(found, vec![3, 1]);
        }
        Err(other) => panic!("expected a shape mismatch, got {other:?}"),
        Ok(_) => panic!("the mismatched centering vector was accepted"),
    }
}
