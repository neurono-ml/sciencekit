//! TDD tests for the sparse×dense and dense×sparse pairwise paths.

use ndarray::Array2;

use super::super::{
    sk_chebyshev_distance_matrix, sk_cosine_distance_matrix, sk_euclidean_distance_matrix,
    sk_manhattan_distance_matrix, sk_squared_euclidean_distance_matrix,
};
use super::chebyshev::{
    sk_dense_sparse_chebyshev_distance_matrix, sk_sparse_dense_chebyshev_distance_matrix,
};
use super::cosine::{
    sk_dense_sparse_cosine_distance_matrix, sk_sparse_dense_cosine_distance_matrix,
};
use super::fixtures::{assert_matrix_close, csr_from_dense, dense_fixture};
use super::manhattan::{
    sk_dense_sparse_manhattan_distance_matrix, sk_sparse_dense_manhattan_distance_matrix,
};
use super::squared_euclidean::{
    sk_dense_sparse_euclidean_distance_matrix, sk_dense_sparse_squared_euclidean_distance_matrix,
    sk_sparse_dense_euclidean_distance_matrix, sk_sparse_dense_squared_euclidean_distance_matrix,
};

/// A dense right fixture with a different row count from the left fixture.
fn right_fixture() -> Array2<f64> {
    ndarray::array![[0.0, 2.0, 0.0, -1.0], [1.0, 0.0, 0.0, 0.0]]
}

/// Squared Euclidean mixed paths equal the dense kernel.
#[test]
fn sparse_squared_euclidean_mixed_paths_match_dense() {
    let left_dense = dense_fixture();
    let right_dense = right_fixture();
    let left_sparse = csr_from_dense(&left_dense);
    let right_sparse = csr_from_dense(&right_dense);
    let expected = sk_squared_euclidean_distance_matrix(&left_dense.view(), &right_dense.view());
    let sparse_dense =
        sk_sparse_dense_squared_euclidean_distance_matrix(&left_sparse.view(), &right_dense.view());
    assert_matrix_close(&sparse_dense, &expected, 1e-12);
    let dense_sparse =
        sk_dense_sparse_squared_euclidean_distance_matrix(&left_dense.view(), &right_sparse.view());
    assert_matrix_close(&dense_sparse, &expected, 1e-12);
}

/// Euclidean mixed paths equal the dense kernel.
#[test]
fn sparse_euclidean_mixed_paths_match_dense() {
    let left_dense = dense_fixture();
    let right_dense = right_fixture();
    let left_sparse = csr_from_dense(&left_dense);
    let right_sparse = csr_from_dense(&right_dense);
    let expected = sk_euclidean_distance_matrix(&left_dense.view(), &right_dense.view());
    let sparse_dense =
        sk_sparse_dense_euclidean_distance_matrix(&left_sparse.view(), &right_dense.view());
    assert_matrix_close(&sparse_dense, &expected, 1e-12);
    let dense_sparse =
        sk_dense_sparse_euclidean_distance_matrix(&left_dense.view(), &right_sparse.view());
    assert_matrix_close(&dense_sparse, &expected, 1e-12);
}

/// Manhattan mixed paths equal the dense kernel.
#[test]
fn sparse_manhattan_mixed_paths_match_dense() {
    let left_dense = dense_fixture();
    let right_dense = right_fixture();
    let left_sparse = csr_from_dense(&left_dense);
    let right_sparse = csr_from_dense(&right_dense);
    let expected = sk_manhattan_distance_matrix(&left_dense.view(), &right_dense.view());
    let sparse_dense =
        sk_sparse_dense_manhattan_distance_matrix(&left_sparse.view(), &right_dense.view());
    assert_matrix_close(&sparse_dense, &expected, 1e-12);
    let dense_sparse =
        sk_dense_sparse_manhattan_distance_matrix(&left_dense.view(), &right_sparse.view());
    assert_matrix_close(&dense_sparse, &expected, 1e-12);
}

/// Chebyshev mixed paths equal the dense kernel.
#[test]
fn sparse_chebyshev_mixed_paths_match_dense() {
    let left_dense = dense_fixture();
    let right_dense = right_fixture();
    let left_sparse = csr_from_dense(&left_dense);
    let right_sparse = csr_from_dense(&right_dense);
    let expected = sk_chebyshev_distance_matrix(&left_dense.view(), &right_dense.view());
    let sparse_dense =
        sk_sparse_dense_chebyshev_distance_matrix(&left_sparse.view(), &right_dense.view());
    assert_matrix_close(&sparse_dense, &expected, 1e-12);
    let dense_sparse =
        sk_dense_sparse_chebyshev_distance_matrix(&left_dense.view(), &right_sparse.view());
    assert_matrix_close(&dense_sparse, &expected, 1e-12);
}

/// Cosine mixed paths equal the dense kernel.
#[test]
fn sparse_cosine_mixed_paths_match_dense() {
    let left_dense = dense_fixture();
    let right_dense = right_fixture();
    let left_sparse = csr_from_dense(&left_dense);
    let right_sparse = csr_from_dense(&right_dense);
    let expected = sk_cosine_distance_matrix(&left_dense.view(), &right_dense.view());
    let sparse_dense =
        sk_sparse_dense_cosine_distance_matrix(&left_sparse.view(), &right_dense.view());
    assert_matrix_close(&sparse_dense, &expected, 1e-12);
    let dense_sparse =
        sk_dense_sparse_cosine_distance_matrix(&left_dense.view(), &right_sparse.view());
    assert_matrix_close(&dense_sparse, &expected, 1e-12);
}
