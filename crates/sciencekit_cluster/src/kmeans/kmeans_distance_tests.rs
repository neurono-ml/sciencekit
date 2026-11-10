//! Companion tests for the squared-Euclidean assignment kernels.

use ndarray::{Array2, array};
use sciencekit_common::{SKDataView, SKError};
use sprs::{CsMat, TriMat};

use super::sk_squared_distances_to_centers;

fn dense_fixture() -> Array2<f64> {
    array![[0.0_f64, 0.0], [1.0, 0.0], [0.0, 2.0]]
}

fn center_fixture() -> Array2<f64> {
    array![[0.0_f64, 0.0], [1.0, 1.0]]
}

fn csr_from_dense(dense: &Array2<f64>) -> CsMat<f64> {
    let mut triplet = TriMat::new(dense.dim());
    for ((row, column), &value) in dense.indexed_iter() {
        if value != 0.0 {
            triplet.add_triplet(row, column, value);
        }
    }
    triplet.to_csr()
}

#[test]
fn kmeans_distance_dense_matches_hand_computed_squared_distances() {
    let distances = sk_squared_distances_to_centers(
        SKDataView::Dense(dense_fixture().view()),
        &center_fixture().view(),
    )
    .unwrap();

    let expected = array![[0.0_f64, 2.0], [1.0, 1.0], [4.0, 2.0]];
    assert_eq!(distances, expected);
}

#[test]
fn kmeans_distance_sparse_matches_dense() {
    let dense = dense_fixture();
    let sparse = csr_from_dense(&dense);
    let centers = center_fixture();

    let dense_distances =
        sk_squared_distances_to_centers(SKDataView::Dense(dense.view()), &centers.view()).unwrap();
    let sparse_distances =
        sk_squared_distances_to_centers(SKDataView::Sparse(sparse.view()), &centers.view()).unwrap();

    for (dense_value, sparse_value) in dense_distances.iter().zip(sparse_distances.iter()) {
        assert!((dense_value - sparse_value).abs() < 1e-12);
    }
}

#[test]
fn kmeans_distance_sparse_wide_design_is_not_densified() {
    let width = 1_000_000;
    let mut triplet = TriMat::new((1, width));
    triplet.add_triplet(0, 0, 2.0);
    triplet.add_triplet(0, width - 1, 3.0);
    let sparse = triplet.to_csr();
    let nnz_before = sparse.nnz();

    let mut centers = Array2::<f64>::zeros((1, width));
    centers[[0, 0]] = 2.0;
    centers[[0, width - 1]] = 3.0;

    let distances =
        sk_squared_distances_to_centers(SKDataView::Sparse(sparse.view()), &centers.view()).unwrap();
    assert!(distances[[0, 0]].abs() < 1e-12);
    assert_eq!(sparse.nnz(), nnz_before);
}

#[test]
fn kmeans_distance_rejects_feature_count_mismatch() {
    let centers = array![[0.0_f64, 0.0, 0.0]];
    let error = sk_squared_distances_to_centers(
        SKDataView::Dense(dense_fixture().view()),
        &centers.view(),
    )
    .unwrap_err();

    match error {
        SKError::ShapeMismatch { expected, found } => {
            assert_eq!(expected, vec![3, 3]);
            assert_eq!(found, vec![3, 2]);
        }
        other => panic!("expected ShapeMismatch, found {other:?}"),
    }
}
