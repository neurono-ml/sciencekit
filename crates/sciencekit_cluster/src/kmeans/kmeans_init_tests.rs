//! Companion tests for seeded centroid initialization.

use ndarray::{Array2, array};
use sciencekit_common::SKDataView;
use sprs::{CsMat, TriMat};

use super::sk_initialize_centers;
use crate::kmeans::SKKMeansInitialization;

fn blobs() -> Array2<f64> {
    array![
        [0.0_f64, 0.0],
        [0.2, 0.1],
        [10.0, 10.0],
        [10.2, 9.9],
        [0.1, 0.2],
        [9.9, 10.1]
    ]
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

fn assert_centers_distinct(centers: &Array2<f64>) {
    for left in 0..centers.nrows() {
        for right in (left + 1)..centers.nrows() {
            assert_ne!(centers.row(left), centers.row(right));
        }
    }
}

#[test]
fn kmeans_init_plus_plus_centers_are_distinct() {
    let data = blobs();
    let centers = sk_initialize_centers(
        SKDataView::Dense(data.view()),
        4,
        SKKMeansInitialization::KMeansPlusPlus,
        42,
    )
    .unwrap();

    assert_eq!(centers.dim(), (4, 2));
    assert_centers_distinct(&centers);
}

#[test]
fn kmeans_init_plus_plus_is_reproducible() {
    let data = blobs();
    let first = sk_initialize_centers(
        SKDataView::Dense(data.view()),
        3,
        SKKMeansInitialization::KMeansPlusPlus,
        7,
    )
    .unwrap();
    let second = sk_initialize_centers(
        SKDataView::Dense(data.view()),
        3,
        SKKMeansInitialization::KMeansPlusPlus,
        7,
    )
    .unwrap();

    assert_eq!(first, second);
}

#[test]
fn kmeans_init_random_is_reproducible_and_distinct() {
    let data = blobs();
    let first = sk_initialize_centers(
        SKDataView::Dense(data.view()),
        3,
        SKKMeansInitialization::Random,
        11,
    )
    .unwrap();
    let second = sk_initialize_centers(
        SKDataView::Dense(data.view()),
        3,
        SKKMeansInitialization::Random,
        11,
    )
    .unwrap();

    assert_eq!(first, second);
    assert_centers_distinct(&first);
}

#[test]
fn kmeans_init_sparse_matches_dense() {
    let dense = blobs();
    let sparse = csr_from_dense(&dense);
    for initialization in [
        SKKMeansInitialization::KMeansPlusPlus,
        SKKMeansInitialization::Random,
    ] {
        let dense_centers =
            sk_initialize_centers(SKDataView::Dense(dense.view()), 3, initialization, 5).unwrap();
        let sparse_centers =
            sk_initialize_centers(SKDataView::Sparse(sparse.view()), 3, initialization, 5).unwrap();
        for (dense_value, sparse_value) in dense_centers.iter().zip(sparse_centers.iter()) {
            assert!((dense_value - sparse_value).abs() < 1e-12);
        }
    }
}
