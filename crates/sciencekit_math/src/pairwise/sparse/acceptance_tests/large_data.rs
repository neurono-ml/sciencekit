//! Acceptance: at least 10 000 sparse rows across every metric.

use ndarray::Array2;
use sciencekit_common::SKDataView;

use crate::pairwise::sparse::fixtures::{assert_matrix_close, large_sparse_fixture};
use crate::pairwise::{SKPairwiseMetric, sk_pairwise_distance_matrix};

/// Every metric on 10 000 sparse left rows, in the sparse-containing cells.
#[test]
fn sparse_pairwise_acceptance_large_data() {
    let left_sparse = large_sparse_fixture(10_000, 32);
    let right_sparse = large_sparse_fixture(16, 32);
    let left_dense: Array2<f64> = left_sparse.to_dense();
    let right_dense: Array2<f64> = right_sparse.to_dense();
    let metrics = [
        SKPairwiseMetric::SquaredEuclidean,
        SKPairwiseMetric::Euclidean,
        SKPairwiseMetric::Manhattan,
        SKPairwiseMetric::Chebyshev,
        SKPairwiseMetric::Cosine,
    ];
    for metric in metrics {
        let expected = sk_pairwise_distance_matrix(
            metric,
            SKDataView::Dense(left_dense.view()),
            SKDataView::Dense(right_dense.view()),
        )
        .expect("dense reference");
        let cells = [
            (
                SKDataView::Sparse(left_sparse.view()),
                SKDataView::Sparse(right_sparse.view()),
                "sparse×sparse",
            ),
            (
                SKDataView::Sparse(left_sparse.view()),
                SKDataView::Dense(right_dense.view()),
                "sparse×dense",
            ),
            (
                SKDataView::Dense(left_dense.view()),
                SKDataView::Sparse(right_sparse.view()),
                "dense×sparse",
            ),
        ];
        for (left, right, label) in cells {
            let result = sk_pairwise_distance_matrix(metric, left, right)
                .unwrap_or_else(|error| panic!("{metric:?} {label}: {error:?}"));
            assert_eq!(result.shape(), &[10_000, 16], "{metric:?} {label} shape");
            assert_matrix_close(&result, &expected, 1e-10);
        }
    }
}
