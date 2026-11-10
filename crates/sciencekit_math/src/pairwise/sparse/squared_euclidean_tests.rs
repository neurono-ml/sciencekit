//! TDD tests for the sparse squared-Euclidean and Euclidean kernels.

use ndarray::{Array2, array};
use sprs::TriMat;

use super::super::{sk_euclidean_distance_matrix, sk_squared_euclidean_distance_matrix};
use super::fixtures::{assert_matrix_close, csr_from_dense, csr_from_triplets, dense_fixture};
use super::{sk_sparse_euclidean_distance_matrix, sk_sparse_squared_euclidean_distance_matrix};

/// Sparse squared Euclidean and Euclidean equal the dense kernels on the same
/// logical matrix.
#[test]
fn sparse_squared_euclidean_matches_dense_on_the_same_logical_matrix() {
    let dense = dense_fixture();
    let sparse = csr_from_dense(&dense);
    let squared = sk_sparse_squared_euclidean_distance_matrix(&sparse.view(), &sparse.view());
    let expected_squared = sk_squared_euclidean_distance_matrix(&dense.view(), &dense.view());
    assert_matrix_close(&squared, &expected_squared, 1e-12);

    let euclidean = sk_sparse_euclidean_distance_matrix(&sparse.view(), &sparse.view());
    let expected_euclidean = sk_euclidean_distance_matrix(&dense.view(), &dense.view());
    assert_matrix_close(&euclidean, &expected_euclidean, 1e-12);
}

/// A sparse row pair with no stored entries is at zero distance.
#[test]
fn sparse_squared_euclidean_empty_rows_have_zero_distance() {
    let sparse = csr_from_triplets(3, 2, &[(2, 1, 3.0)]);
    let squared = sk_sparse_squared_euclidean_distance_matrix(&sparse.view(), &sparse.view());
    assert_eq!(squared[[0, 0]], 0.0);
    assert_eq!(squared[[0, 1]], 0.0);
    assert_eq!(squared[[1, 0]], 0.0);
    let euclidean = sk_sparse_euclidean_distance_matrix(&sparse.view(), &sparse.view());
    assert_eq!(euclidean[[0, 0]], 0.0);
}

/// A sparse set compared with itself is symmetric, non-negative and has a zero
/// diagonal.
#[test]
fn sparse_squared_euclidean_is_symmetric_with_zero_diagonal() {
    let dense = dense_fixture();
    let sparse = csr_from_dense(&dense);
    let squared = sk_sparse_squared_euclidean_distance_matrix(&sparse.view(), &sparse.view());
    for (index, value) in squared.indexed_iter() {
        assert!(*value >= 0.0, "negative entry at {index:?}");
        assert_eq!(*value, squared[[index.1, index.0]]);
    }
    assert_eq!(
        squared.diag(),
        ndarray::Array1::<f64>::zeros(squared.nrows()).view()
    );
}

/// Euclidean is the element-wise square root of the squared form.
#[test]
fn sparse_euclidean_is_sqrt_of_squared() {
    let dense = dense_fixture();
    let sparse = csr_from_dense(&dense);
    let squared = sk_sparse_squared_euclidean_distance_matrix(&sparse.view(), &sparse.view());
    let euclidean = sk_sparse_euclidean_distance_matrix(&sparse.view(), &sparse.view());
    for (index, value) in euclidean.indexed_iter() {
        assert!((*value - squared[index].sqrt()).abs() < 1e-12);
    }
}

/// The sparse kernels accept `f32` as well.
#[test]
fn sparse_squared_euclidean_accepts_f32() {
    let mut triplets = TriMat::new((2, 2));
    triplets.add_triplet(0, 0, 3.0_f32);
    triplets.add_triplet(1, 1, 4.0_f32);
    let sparse = triplets.to_csr();
    let distance = sk_sparse_squared_euclidean_distance_matrix(&sparse.view(), &sparse.view());
    assert!((distance[[0, 1]] - 25.0_f32).abs() < 1e-5);
}

/// The sparse path agrees with the dense path on a rectangular pair.
#[test]
fn sparse_squared_euclidean_matches_dense_on_rectangular_inputs() {
    let left_dense: Array2<f64> = array![[1.0, 0.0, 2.0], [0.0, 3.0, 0.0]];
    let right_dense: Array2<f64> = array![[0.0, 1.5, 0.0]];
    let left = csr_from_dense(&left_dense);
    let right = csr_from_dense(&right_dense);
    let distance = sk_sparse_squared_euclidean_distance_matrix(&left.view(), &right.view());
    let expected = sk_squared_euclidean_distance_matrix(&left_dense.view(), &right_dense.view());
    assert_matrix_close(&distance, &expected, 1e-12);
}
