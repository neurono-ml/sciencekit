//! TDD tests for the sparse Manhattan kernel.

use ndarray::array;

use super::super::sk_manhattan_distance_matrix;
use super::fixtures::{assert_matrix_close, csr_from_dense, csr_from_triplets, dense_fixture};
use super::sk_sparse_manhattan_distance_matrix;

/// Sparse Manhattan equals the dense kernel on the same logical matrix.
#[test]
fn sparse_manhattan_matches_dense_on_the_same_logical_matrix() {
    let dense = dense_fixture();
    let sparse = csr_from_dense(&dense);
    let distance = sk_sparse_manhattan_distance_matrix(&sparse.view(), &sparse.view());
    let expected = sk_manhattan_distance_matrix(&dense.view(), &dense.view());
    assert_matrix_close(&distance, &expected, 1e-12);
}

/// One-sided stored coordinates contribute their absolute value.
#[test]
fn sparse_manhattan_one_sided_coordinates_contribute_absolute_values() {
    let left = csr_from_triplets(1, 3, &[(0, 0, -4.0)]);
    let right = csr_from_triplets(1, 3, &[(0, 2, 2.5)]);
    let distance = sk_sparse_manhattan_distance_matrix(&left.view(), &right.view());
    assert!((distance[[0, 0]] - 6.5).abs() < 1e-12);
}

/// Identical sparse rows are at zero distance.
#[test]
fn sparse_manhattan_identical_rows_have_zero_distance() {
    let sparse = csr_from_triplets(2, 4, &[(0, 0, 1.0), (0, 2, -2.0), (1, 3, 5.0)]);
    let distance = sk_sparse_manhattan_distance_matrix(&sparse.view(), &sparse.view());
    assert_eq!(distance.diag(), ndarray::Array1::zeros(2).view());
}

/// The kernel handles rectangular sparse inputs.
#[test]
fn sparse_manhattan_matches_dense_on_rectangular_inputs() {
    let left_dense = array![[0.0, 2.0], [3.0, 0.0]];
    let right_dense = array![[1.0, 0.0]];
    let left = csr_from_dense(&left_dense);
    let right = csr_from_dense(&right_dense);
    let distance = sk_sparse_manhattan_distance_matrix(&left.view(), &right.view());
    let expected = sk_manhattan_distance_matrix(&left_dense.view(), &right_dense.view());
    assert_matrix_close(&distance, &expected, 1e-12);
}
