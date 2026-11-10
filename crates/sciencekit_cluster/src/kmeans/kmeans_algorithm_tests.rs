//! Companion tests for automatic algorithm resolution.

use ndarray::{Array2, array};
use sciencekit_common::SKDataView;
use sprs::{CsMat, TriMat};

use super::algorithm::{sk_fit_with_algorithm, sk_resolve_algorithm};
use super::{SKKMeansAlgorithm, SKKMeansInitialization, sk_initialize_centers};

fn separated_blobs() -> Array2<f64> {
    array![
        [0.0_f64, 0.0],
        [0.1, 0.0],
        [0.0, 0.1],
        [10.0, 10.0],
        [10.1, 10.0],
        [10.0, 10.1]
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

#[test]
fn kmeans_algorithm_auto_resolves_elkan_for_dense_and_lloyd_for_sparse() {
    let dense = separated_blobs();
    let sparse = csr_from_dense(&dense);

    assert_eq!(
        sk_resolve_algorithm(SKKMeansAlgorithm::Auto, &SKDataView::Dense(dense.view())),
        SKKMeansAlgorithm::Elkan
    );
    assert_eq!(
        sk_resolve_algorithm(SKKMeansAlgorithm::Auto, &SKDataView::Sparse(sparse.view())),
        SKKMeansAlgorithm::Lloyd
    );
}

#[test]
fn kmeans_algorithm_explicit_choice_is_reported() {
    let dense = separated_blobs();
    let sparse = csr_from_dense(&dense);

    assert_eq!(
        sk_resolve_algorithm(SKKMeansAlgorithm::Lloyd, &SKDataView::Dense(dense.view())),
        SKKMeansAlgorithm::Lloyd
    );
    assert_eq!(
        sk_resolve_algorithm(SKKMeansAlgorithm::Elkan, &SKDataView::Dense(dense.view())),
        SKKMeansAlgorithm::Elkan
    );
    assert_eq!(
        sk_resolve_algorithm(SKKMeansAlgorithm::Elkan, &SKDataView::Sparse(sparse.view())),
        SKKMeansAlgorithm::Lloyd
    );
}

#[test]
fn kmeans_algorithm_auto_matches_explicit_lloyd_on_dense_data() {
    let dense = separated_blobs();
    let centers = sk_initialize_centers(
        SKDataView::Dense(dense.view()),
        2,
        SKKMeansInitialization::KMeansPlusPlus,
        8,
    )
    .unwrap();

    let (automatic, automatic_algorithm) = sk_fit_with_algorithm(
        SKDataView::Dense(dense.view()),
        centers.clone(),
        SKKMeansAlgorithm::Auto,
        100,
        1e-9,
    )
    .unwrap();
    let (lloyd, lloyd_algorithm) = sk_fit_with_algorithm(
        SKDataView::Dense(dense.view()),
        centers,
        SKKMeansAlgorithm::Lloyd,
        100,
        1e-9,
    )
    .unwrap();

    assert_eq!(automatic_algorithm, SKKMeansAlgorithm::Elkan);
    assert_eq!(lloyd_algorithm, SKKMeansAlgorithm::Lloyd);
    assert_eq!(automatic.labels, lloyd.labels);
    for (automatic_value, lloyd_value) in automatic.centers.iter().zip(lloyd.centers.iter()) {
        assert!((automatic_value - lloyd_value).abs() < 1e-9);
    }
    assert!((automatic.inertia - lloyd.inertia).abs() < 1e-9);
}

#[test]
fn kmeans_algorithm_auto_sparse_matches_dense_within_tolerance() {
    let dense = separated_blobs();
    let sparse = csr_from_dense(&dense);
    let dense_centers = sk_initialize_centers(
        SKDataView::Dense(dense.view()),
        2,
        SKKMeansInitialization::KMeansPlusPlus,
        13,
    )
    .unwrap();
    let sparse_centers = sk_initialize_centers(
        SKDataView::Sparse(sparse.view()),
        2,
        SKKMeansInitialization::KMeansPlusPlus,
        13,
    )
    .unwrap();

    let (dense_result, dense_algorithm) = sk_fit_with_algorithm(
        SKDataView::Dense(dense.view()),
        dense_centers,
        SKKMeansAlgorithm::Auto,
        100,
        1e-9,
    )
    .unwrap();
    let (sparse_result, sparse_algorithm) = sk_fit_with_algorithm(
        SKDataView::Sparse(sparse.view()),
        sparse_centers,
        SKKMeansAlgorithm::Auto,
        100,
        1e-9,
    )
    .unwrap();

    assert_eq!(dense_algorithm, SKKMeansAlgorithm::Elkan);
    assert_eq!(sparse_algorithm, SKKMeansAlgorithm::Lloyd);
    assert_eq!(dense_result.labels, sparse_result.labels);
    assert!((dense_result.inertia - sparse_result.inertia).abs() < 1e-9);
}
