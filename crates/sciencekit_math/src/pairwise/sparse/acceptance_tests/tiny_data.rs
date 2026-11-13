//! Acceptance: tiny sparse data, including empty rows and a single entry.

use ndarray::Array2;
use sciencekit_common::SKDataView;

use crate::pairwise::sparse::fixtures::{assert_matrix_close, csr_from_dense};
use crate::pairwise::{SKPairwiseMetric, sk_pairwise_distance_matrix};

/// Every metric on tiny sparse data with empty rows and one stored entry.
#[test]
fn sparse_pairwise_acceptance_tiny_data() {
    // Rows: empty, a single stored entry, an ordinary row, another empty row.
    let left_dense: Array2<f64> = ndarray::array![
        [0.0, 0.0, 0.0],
        [0.0, 2.5, 0.0],
        [0.0, -3.0, 5.0],
        [0.0, 0.0, 0.0],
    ];
    let right_dense: Array2<f64> = ndarray::array![[0.0, 0.0, 0.0], [1.0, 0.0, -2.0]];
    let left_sparse = csr_from_dense(&left_dense);
    let right_sparse = csr_from_dense(&right_dense);
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
            assert_eq!(result.shape(), &[4, 2], "{metric:?} {label} shape");
            assert_matrix_close(&result, &expected, 1e-12);
        }
    }
}
