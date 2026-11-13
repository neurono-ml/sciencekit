//! TDD tests for the sparse cosine kernel.

use ndarray::array;

use super::super::sk_cosine_distance_matrix;
use super::fixtures::{assert_matrix_close, csr_from_dense, csr_from_triplets, dense_fixture};
use super::sk_sparse_cosine_distance_matrix;

/// Sparse cosine equals the dense kernel on the same logical matrix.
#[test]
fn sparse_cosine_matches_dense_on_the_same_logical_matrix() {
    let dense = dense_fixture();
    let sparse = csr_from_dense(&dense);
    let distance = sk_sparse_cosine_distance_matrix(&sparse.view(), &sparse.view());
    let expected = sk_cosine_distance_matrix(&dense.view(), &dense.view());
    assert_matrix_close(&distance, &expected, 1e-12);
}

/// Scaling a sparse row by a positive scalar leaves its cosine distance
/// unchanged.
#[test]
fn sparse_cosine_is_scale_invariant() {
    let reference = csr_from_triplets(1, 3, &[(0, 0, 1.0), (0, 2, 2.0)]);
    let scaled = csr_from_triplets(1, 3, &[(0, 0, 3.0), (0, 2, 6.0)]);
    let unscaled_distance = sk_sparse_cosine_distance_matrix(&reference.view(), &reference.view());
    let scaled_distance = sk_sparse_cosine_distance_matrix(&scaled.view(), &reference.view());
    assert!((scaled_distance[[0, 0]] - unscaled_distance[[0, 0]]).abs() < 1e-12);
}

/// A zero-norm sparse row yields distance one, as in the dense kernel.
#[test]
fn sparse_cosine_zero_norm_row_keeps_the_dense_policy() {
    let sparse = csr_from_triplets(2, 3, &[(1, 1, 4.0)]);
    let distance = sk_sparse_cosine_distance_matrix(&sparse.view(), &sparse.view());
    assert!((distance[[0, 0]] - 1.0).abs() < 1e-12);
    assert!((distance[[0, 1]] - 1.0).abs() < 1e-12);
    assert!((distance[[1, 0]] - 1.0).abs() < 1e-12);
    assert!(distance[[1, 1]].abs() < 1e-12);
}

/// Orthogonal sparse rows are at distance one.
#[test]
fn sparse_cosine_orthogonal_rows_have_unit_distance() {
    let left = csr_from_triplets(1, 3, &[(0, 0, 2.0)]);
    let right = csr_from_triplets(1, 3, &[(0, 1, 5.0)]);
    let distance = sk_sparse_cosine_distance_matrix(&left.view(), &right.view());
    assert!((distance[[0, 0]] - 1.0).abs() < 1e-12);
}

/// The kernel handles rectangular sparse inputs.
#[test]
fn sparse_cosine_matches_dense_on_rectangular_inputs() {
    let left_dense = array![[2.0, 0.0], [0.0, -3.0]];
    let right_dense = array![[1.0, 1.0]];
    let left = csr_from_dense(&left_dense);
    let right = csr_from_dense(&right_dense);
    let distance = sk_sparse_cosine_distance_matrix(&left.view(), &right.view());
    let expected = sk_cosine_distance_matrix(&left_dense.view(), &right_dense.view());
    assert_matrix_close(&distance, &expected, 1e-12);
}
