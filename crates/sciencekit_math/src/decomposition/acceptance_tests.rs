//! Acceptance tests of the randomized range finder (PRD §8.7).
//!
//! The suite runs the kernel with lots and little data, over dense and sparse
//! representations, and pins the documented accuracy tolerance against the
//! exact singular-value decomposition.

use ndarray::{Array1, Array2};
use sciencekit_common::SKDataView;

use super::dense_tests::{fixture_low_rank_matrix, fixture_matrix, relative_error};
use super::qr::sk_thin_qr;
use super::randomized::sk_randomized_svd;
use super::sparse_tests::to_csr;
use crate::backend::{SKMathBackend, SKFaerBackend};

/// The documented tolerance of the singular values against the exact
/// decomposition.
const SINGULAR_VALUE_TOLERANCE: f64 = 1e-6;

/// The documented tolerance of the reconstruction against the exact
/// decomposition.
const RECONSTRUCTION_TOLERANCE: f64 = 1e-6;

/// The relative Frobenius error of the best rank-`k` approximation of a matrix
/// with the given exact singular values.
fn expected_truncation_error(singular_values: &[f64], components: usize) -> f64 {
    let tail: f64 = singular_values
        .iter()
        .skip(components)
        .map(|value| value * value)
        .sum();
    let total: f64 = singular_values.iter().map(|value| value * value).sum();
    (tail / total).sqrt()
}

/// A fixture with an exactly known, decaying spectrum.
fn decaying_spectrum_fixture(rows: usize, columns: usize, decay: f64) -> Array2<f64> {
    let rank = rows.min(columns);
    let left = sk_thin_qr(&fixture_matrix(rows, rank, 3).view());
    let right = sk_thin_qr(&fixture_matrix(columns, rank, 5).view());
    let mut result = Array2::<f64>::zeros((rows, columns));
    for index in 0..rank {
        let singular_value = (-(decay * index as f64)).exp();
        for row in 0..rows {
            for column in 0..columns {
                result[(row, column)] +=
                    singular_value * left[(row, index)] * right[(column, index)];
            }
        }
    }
    result
}

/// The exact singular values of a matrix, computed with the backend.
fn exact_singular_values(matrix: &Array2<f64>) -> Vec<f64> {
    let backend = SKFaerBackend::new();
    let decomposition = backend
        .svd(matrix.view())
        .expect("the exact decomposition must succeed");
    decomposition.singular_values
}

#[test]
fn randomized_acceptance_tiny_matrices_are_factorized() {
    let single = Array2::<f64>::from_shape_vec((1, 1), vec![4.0]).unwrap();
    let result = sk_randomized_svd(SKDataView::Dense(single.view()), 1, 1, 1, 0, None)
        .expect("the decomposition must succeed");
    assert_eq!(result.u.dim(), (1, 1));
    assert_eq!(result.v.dim(), (1, 1));
    assert!((result.singular_values[0] - 4.0).abs() < 1e-8);

    let tiny = fixture_matrix(3, 2, 11);
    let result = sk_randomized_svd(SKDataView::Dense(tiny.view()), 2, 1, 1, 0, None)
        .expect("the decomposition must succeed");
    assert_eq!(result.u.dim(), (3, 2));
    assert_eq!(result.v.dim(), (2, 2));
}

#[test]
fn randomized_acceptance_every_component_count_boundary_is_supported() {
    let matrix = fixture_low_rank_matrix(20, 8, 4, 61);
    for components in 1..=8usize {
        let result = sk_randomized_svd(SKDataView::Dense(matrix.view()), components, 2, 3, 5, None)
            .expect("the decomposition must succeed");
        assert_eq!(result.u.dim(), (20, components));
        assert_eq!(result.v.dim(), (8, components));
        assert_eq!(result.singular_values.len(), components);
    }
}

#[test]
fn randomized_acceptance_singular_values_match_the_exact_decomposition() {
    let matrix = decaying_spectrum_fixture(60, 30, 0.4);
    let components = 6;
    let result = sk_randomized_svd(SKDataView::Dense(matrix.view()), components, 7, 10, 13, None)
        .expect("the decomposition must succeed");
    let exact = exact_singular_values(&matrix);

    for (index, (&approximate, &expected)) in result
        .singular_values
        .iter()
        .zip(exact.iter())
        .enumerate()
    {
        assert!(
            (approximate - expected).abs() <= SINGULAR_VALUE_TOLERANCE * expected.max(1.0),
            "the singular value {index} differs: {approximate} against {expected}"
        );
    }
}

#[test]
fn randomized_acceptance_reconstruction_matches_the_exact_decomposition() {
    let matrix = decaying_spectrum_fixture(60, 30, 0.4);
    let components = 6;
    let result = sk_randomized_svd(SKDataView::Dense(matrix.view()), components, 7, 10, 13, None)
        .expect("the decomposition must succeed");
    let error = relative_error(&matrix, &result.reconstruct());
    let exact = exact_singular_values(&matrix);
    let expected = expected_truncation_error(&exact, components);
    assert!(
        error <= expected + RECONSTRUCTION_TOLERANCE,
        "the reconstruction error {error} exceeds the optimal truncation error {expected}"
    );
}

#[test]
fn randomized_acceptance_a_decaying_spectrum_is_recovered() {
    let matrix = decaying_spectrum_fixture(80, 40, 0.5);
    let result = sk_randomized_svd(SKDataView::Dense(matrix.view()), 5, 5, 10, 17, None)
        .expect("the decomposition must succeed");
    let exact = exact_singular_values(&matrix);
    for (index, (&approximate, &expected)) in result
        .singular_values
        .iter()
        .zip(exact.iter())
        .enumerate()
    {
        assert!(
            (approximate - expected).abs() <= SINGULAR_VALUE_TOLERANCE * expected.max(1.0),
            "the singular value {index} differs: {approximate} against {expected}"
        );
    }
}

#[test]
fn randomized_acceptance_large_dense_matrix_is_factorized() {
    let matrix = fixture_low_rank_matrix(600, 200, 10, 71);
    let result = sk_randomized_svd(SKDataView::Dense(matrix.view()), 8, 3, 10, 19, None)
        .expect("the decomposition must succeed");
    assert_eq!(result.u.dim(), (600, 8));
    assert_eq!(result.v.dim(), (200, 8));
    let error = relative_error(&matrix, &result.reconstruct());
    let exact = exact_singular_values(&matrix);
    let expected = expected_truncation_error(&exact, 8);
    assert!(
        error <= expected + RECONSTRUCTION_TOLERANCE,
        "the reconstruction error {error} exceeds the optimal truncation error {expected}"
    );
}

#[test]
fn randomized_acceptance_large_sparse_matrix_is_factorized() {
    let mut dense = Array2::<f64>::zeros((500, 1_000));
    for row in 0..500 {
        for offset in 0..4usize {
            dense[(row, (row * 3 + offset) % 1_000)] = 1.0 + offset as f64;
        }
    }
    let sparse = to_csr(&dense);
    let result = sk_randomized_svd(SKDataView::Sparse(sparse.view()), 6, 3, 8, 23, None)
        .expect("the decomposition must succeed");
    assert_eq!(result.u.dim(), (500, 6));
    assert_eq!(result.v.dim(), (1_000, 6));
    assert!(result.singular_values.iter().all(|value| *value >= 0.0));
}

#[test]
fn randomized_acceptance_produces_the_reconstruction_error_metric() {
    let matrix = fixture_low_rank_matrix(40, 12, 3, 73);
    let result = sk_randomized_svd(SKDataView::Dense(matrix.view()), 3, 2, 4, 29, None)
        .expect("the decomposition must succeed");
    let metric = result
        .relative_frobenius_error(&matrix.view())
        .expect("the metric must be computed");
    assert!(metric < 1e-8, "the reconstruction error {metric} is too large");
}

#[test]
fn randomized_acceptance_centered_large_sparse_matrix_is_factorized() {
    let mut dense = Array2::<f64>::zeros((400, 800));
    for row in 0..400 {
        dense[(row, row % 800)] = 2.0;
    }
    let sparse = to_csr(&dense);
    let mean = Array1::<f64>::from_elem(800, 0.05);
    let result = sk_randomized_svd(
        SKDataView::Sparse(sparse.view()),
        5,
        2,
        6,
        31,
        Some(mean.view()),
    )
    .expect("the centered decomposition must succeed");
    assert_eq!(result.u.dim(), (400, 5));
    assert_eq!(result.v.dim(), (800, 5));
}
