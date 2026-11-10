//! TDD tests covering every cell of the metric × representation matrix.

use ndarray::Array2;
use sciencekit_common::{SKDataView, SKError};
use sprs::CsMat;

use super::{
    SKPairwiseMetric, sk_chebyshev_distance_matrix, sk_cosine_distance_matrix,
    sk_euclidean_distance_matrix, sk_manhattan_distance_matrix, sk_pairwise_distance_matrix,
    sk_squared_euclidean_distance_matrix,
};
use crate::pairwise::sparse::fixtures::csr_from_dense;

/// One representation combination with its two views.
fn representation_pairs<'a>(
    left_dense: &'a Array2<f64>,
    right_dense: &'a Array2<f64>,
    left_sparse: &'a CsMat<f64>,
    right_sparse: &'a CsMat<f64>,
) -> Vec<(SKDataView<'a, f64>, SKDataView<'a, f64>)> {
    vec![
        (
            SKDataView::Dense(left_dense.view()),
            SKDataView::Dense(right_dense.view()),
        ),
        (
            SKDataView::Sparse(left_sparse.view()),
            SKDataView::Sparse(right_sparse.view()),
        ),
        (
            SKDataView::Sparse(left_sparse.view()),
            SKDataView::Dense(right_dense.view()),
        ),
        (
            SKDataView::Dense(left_dense.view()),
            SKDataView::Sparse(right_sparse.view()),
        ),
    ]
}

/// Assert every entry of two matrices agrees within the tolerance.
fn assert_close(left: &Array2<f64>, right: &Array2<f64>, label: &str) {
    assert_eq!(left.shape(), right.shape(), "{label} shape");
    for ((index, left_value), right_value) in left.indexed_iter().zip(right.iter()) {
        assert!(
            (left_value - right_value).abs() < 1e-12,
            "{label} mismatch at {index:?}: {left_value} vs {right_value}"
        );
    }
}

/// Every metric matches its dense kernel in all four representation cells.
#[test]
fn sparse_pairwise_dispatches_every_metric_representation_cell() {
    let left_dense = crate::pairwise::sparse::fixtures::dense_fixture();
    let right_dense: Array2<f64> =
        ndarray::array![[0.0, 2.0, 0.0, -1.0], [1.0, 0.0, 0.0, 0.0]];
    let left_sparse = csr_from_dense(&left_dense);
    let right_sparse = csr_from_dense(&right_dense);
    let metrics = [
        (
            SKPairwiseMetric::SquaredEuclidean,
            sk_squared_euclidean_distance_matrix(&left_dense.view(), &right_dense.view()),
        ),
        (
            SKPairwiseMetric::Euclidean,
            sk_euclidean_distance_matrix(&left_dense.view(), &right_dense.view()),
        ),
        (
            SKPairwiseMetric::Manhattan,
            sk_manhattan_distance_matrix(&left_dense.view(), &right_dense.view()),
        ),
        (
            SKPairwiseMetric::Chebyshev,
            sk_chebyshev_distance_matrix(&left_dense.view(), &right_dense.view()),
        ),
        (
            SKPairwiseMetric::Cosine,
            sk_cosine_distance_matrix(&left_dense.view(), &right_dense.view()),
        ),
    ];
    for (metric, expected) in metrics {
        for (cell, (left, right)) in representation_pairs(
            &left_dense,
            &right_dense,
            &left_sparse,
            &right_sparse,
        )
        .into_iter()
        .enumerate()
        {
            let result = sk_pairwise_distance_matrix(metric, left, right)
                .unwrap_or_else(|error| panic!("{metric:?} cell {cell} failed: {error:?}"));
            assert_close(&result, &expected, &format!("{metric:?} cell {cell}"));
        }
    }
}

/// A sparse NaN-aware request is rejected in every mixed cell, and the dense
/// cell keeps working.
#[test]
fn sparse_pairwise_nan_euclidean_is_dense_only_in_every_cell() {
    let left_dense = crate::pairwise::sparse::fixtures::dense_fixture();
    let right_dense = left_dense.clone();
    let left_sparse = csr_from_dense(&left_dense);
    let right_sparse = csr_from_dense(&right_dense);
    for (cell, (left, right)) in representation_pairs(
        &left_dense,
        &right_dense,
        &left_sparse,
        &right_sparse,
    )
    .into_iter()
    .enumerate()
    {
        let outcome =
            sk_pairwise_distance_matrix(SKPairwiseMetric::NanEuclidean, left, right);
        if cell == 0 {
            assert!(outcome.is_ok(), "dense NaN-aware cell must work");
            continue;
        }
        match outcome {
            Err(SKError::UnsupportedRepresentation { representation, .. }) => {
                assert_eq!(representation, "csr", "cell {cell}");
            }
            other => panic!("cell {cell}: expected UnsupportedRepresentation, got {other:?}"),
        }
    }
}
