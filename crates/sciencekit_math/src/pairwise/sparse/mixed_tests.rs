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

/// A wide sparse matrix with few stored entries is computed from its stored
/// values for every metric, through the public dispatcher, without
/// materializing a dense copy (a dense copy of a hundred-million-column matrix
/// would not fit in memory).
#[test]
fn sparse_pairwise_wide_matrix_is_not_densified() {
    let width = 100_000_000;
    let sparse = super::fixtures::csr_from_triplets(
        3,
        width,
        &[(0, 0, 2.0), (0, width - 1, 3.0), (1, 0, 2.0)],
    );
    assert_eq!(sparse.nnz(), 3);
    let view = sciencekit_common::SKDataView::Sparse(sparse.view());
    for metric in [
        crate::pairwise::SKPairwiseMetric::SquaredEuclidean,
        crate::pairwise::SKPairwiseMetric::Euclidean,
        crate::pairwise::SKPairwiseMetric::Manhattan,
        crate::pairwise::SKPairwiseMetric::Chebyshev,
        crate::pairwise::SKPairwiseMetric::Cosine,
    ] {
        let distance = crate::pairwise::sk_pairwise_distance_matrix(metric, view, view).unwrap();
        assert_eq!(distance.dim(), (3, 3));
    }
    assert_eq!(sparse.nnz(), 3);
    let squared =
        super::sk_sparse_squared_euclidean_distance_matrix(&sparse.view(), &sparse.view());
    assert!((squared[[0, 1]] - 9.0).abs() < 1e-12);
    assert!((squared[[0, 2]] - 13.0).abs() < 1e-12);
    assert!((squared[[1, 2]] - 4.0).abs() < 1e-12);
}

/// The Chebyshev maximum may sit at a dense coordinate the sparse row does not
/// store, in both mixed directions.
#[test]
fn sparse_chebyshev_sees_one_sided_dense_maximum() {
    let sparse = super::fixtures::csr_from_triplets(1, 6, &[(0, 0, 1.0)]);
    let dense: Array2<f64> = ndarray::array![[1.0, 0.0, 0.0, 0.0, 0.0, 9.0]];
    let sparse_dense = sk_sparse_dense_chebyshev_distance_matrix(&sparse.view(), &dense.view());
    assert!((sparse_dense[[0, 0]] - 9.0).abs() < 1e-12);
    let dense_sparse = sk_dense_sparse_chebyshev_distance_matrix(&dense.view(), &sparse.view());
    assert!((dense_sparse[[0, 0]] - 9.0).abs() < 1e-12);
}
