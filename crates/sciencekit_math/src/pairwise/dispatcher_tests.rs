//! TDD tests for the pairwise dispatch surface (spec `pairwise-distances`).

use ndarray::{Array2, array};
use sciencekit_common::{SKDataView, SKError};
use sprs::TriMat;

use super::{
    SKPairwiseMetric, sk_chebyshev_distance_matrix, sk_cosine_distance_matrix,
    sk_euclidean_distance_matrix, sk_manhattan_distance_matrix,
    sk_nan_euclidean_distance_matrix, sk_pairwise_distance_matrix,
    sk_squared_euclidean_distance_matrix,
};

/// The dense fixture every metric-selection test compares against.
fn dense_fixture() -> Array2<f64> {
    array![[0.0, 1.0, 2.0], [3.0, 0.0, 4.0], [1.0, 1.0, 1.0]]
}

/// Build a sparse matrix from explicit triplets.
fn sparse_fixture() -> sprs::CsMat<f64> {
    let mut triplets = TriMat::new((2, 3));
    triplets.add_triplet(0, 0, 1.0);
    triplets.add_triplet(0, 2, 3.0);
    triplets.add_triplet(1, 1, 2.0);
    triplets.to_csr()
}

/// Assert two matrices agree within the documented tolerance.
fn assert_matrix_close(left: &Array2<f64>, right: &Array2<f64>) {
    assert_eq!(left.shape(), right.shape());
    for ((index, left_value), right_value) in left.indexed_iter().zip(right.iter()) {
        assert!(
            (left_value - right_value).abs() < 1e-12,
            "mismatch at {index:?}: {left_value} vs {right_value}"
        );
    }
}

/// Every metric selects the matching dense kernel.
#[test]
fn sparse_pairwise_dispatches_each_metric_to_its_dense_kernel() {
    let fixture = dense_fixture();
    let left = SKDataView::Dense(fixture.view());
    let right = SKDataView::Dense(fixture.view());
    let cases = [
        (
            SKPairwiseMetric::SquaredEuclidean,
            sk_squared_euclidean_distance_matrix(&fixture.view(), &fixture.view()),
        ),
        (
            SKPairwiseMetric::Euclidean,
            sk_euclidean_distance_matrix(&fixture.view(), &fixture.view()),
        ),
        (
            SKPairwiseMetric::Manhattan,
            sk_manhattan_distance_matrix(&fixture.view(), &fixture.view()),
        ),
        (
            SKPairwiseMetric::Chebyshev,
            sk_chebyshev_distance_matrix(&fixture.view(), &fixture.view()),
        ),
        (
            SKPairwiseMetric::Cosine,
            sk_cosine_distance_matrix(&fixture.view(), &fixture.view()),
        ),
    ];
    for (metric, expected) in cases {
        let result = sk_pairwise_distance_matrix(metric, left, right).expect("dense dispatch works");
        assert_matrix_close(&result, &expected);
    }
}

/// NaN-aware Euclidean over dense views keeps the dense kernel behaviour.
#[test]
fn sparse_pairwise_dense_nan_euclidean_keeps_the_dense_kernel() {
    let fixture = array![[0.0, f64::NAN, 2.0], [1.0, 1.0, f64::NAN]];
    let result = sk_pairwise_distance_matrix(
        SKPairwiseMetric::NanEuclidean,
        SKDataView::Dense(fixture.view()),
        SKDataView::Dense(fixture.view()),
    )
    .expect("dense NaN-aware dispatch works");
    let expected = sk_nan_euclidean_distance_matrix(&fixture.view(), &fixture.view());
    assert_eq!(result, expected);
}

/// A feature-count mismatch is rejected with both dimensions.
#[test]
fn sparse_pairwise_rejects_feature_count_mismatch() {
    let left: Array2<f64> = Array2::zeros((3, 2));
    let right: Array2<f64> = Array2::zeros((4, 3));
    let error = sk_pairwise_distance_matrix(
        SKPairwiseMetric::Euclidean,
        SKDataView::Dense(left.view()),
        SKDataView::Dense(right.view()),
    )
    .expect_err("mismatched feature counts must be rejected");
    match error {
        SKError::ShapeMismatch { expected, found } => {
            assert_eq!(expected, vec![3, 2]);
            assert_eq!(found, vec![4, 3]);
        }
        other => panic!("expected ShapeMismatch, got {other:?}"),
    }
}

/// Empty inputs produce an empty matrix instead of an error.
#[test]
fn sparse_pairwise_empty_inputs_produce_empty_matrices() {
    let empty: Array2<f64> = Array2::zeros((0, 2));
    let fixture: Array2<f64> = Array2::ones((3, 2));
    let left_empty = sk_pairwise_distance_matrix(
        SKPairwiseMetric::Euclidean,
        SKDataView::Dense(empty.view()),
        SKDataView::Dense(fixture.view()),
    )
    .expect("empty left is valid");
    assert_eq!(left_empty.shape(), &[0, 3]);
    let right_empty = sk_pairwise_distance_matrix(
        SKPairwiseMetric::Manhattan,
        SKDataView::Dense(fixture.view()),
        SKDataView::Dense(empty.view()),
    )
    .expect("empty right is valid");
    assert_eq!(right_empty.shape(), &[3, 0]);
}

/// A sparse request for the NaN-aware Euclidean metric is rejected up front.
#[test]
fn sparse_pairwise_rejects_sparse_nan_euclidean() {
    let sparse = sparse_fixture();
    let dense: Array2<f64> = array![[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]];
    let sparse_left = SKDataView::Sparse(sparse.view());
    let dense_right = SKDataView::Dense(dense.view());
    for (left, right) in [
        (sparse_left, dense_right),
        (dense_right, sparse_left),
    ] {
        let error = sk_pairwise_distance_matrix(SKPairwiseMetric::NanEuclidean, left, right)
            .expect_err("sparse NaN-aware input must be rejected");
        match error {
            SKError::UnsupportedRepresentation {
                representation,
                suggestion,
            } => {
                assert_eq!(representation, "csr");
                assert!(suggestion.contains("dense"), "suggestion: {suggestion}");
            }
            other => panic!("expected UnsupportedRepresentation, got {other:?}"),
        }
    }
}
