//! Acceptance: concurrent sparse distance computation on large data.

use ndarray::Array2;
use sciencekit_common::SKDataView;

use crate::pairwise::sparse::fixtures::large_sparse_fixture;
use crate::pairwise::{SKPairwiseMetric, sk_pairwise_distance_matrix};

/// Several threads computing every metric over the same 10 000-row sparse
/// input reproduce the single-threaded result exactly.
#[test]
fn sparse_pairwise_acceptance_concurrent_execution() {
    let left_sparse = large_sparse_fixture(10_000, 16);
    let right_sparse = large_sparse_fixture(8, 16);
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
            SKDataView::Sparse(left_sparse.view()),
            SKDataView::Sparse(right_sparse.view()),
        )
        .expect("sparse reference");
        let results: Vec<Array2<f64>> = std::thread::scope(|scope| {
            let handles: Vec<_> = (0..4)
                .map(|_| {
                    scope.spawn(|| {
                        sk_pairwise_distance_matrix(
                            metric,
                            SKDataView::Sparse(left_sparse.view()),
                            SKDataView::Sparse(right_sparse.view()),
                        )
                        .expect("concurrent sparse result")
                    })
                })
                .collect();
            handles
                .into_iter()
                .map(|handle| handle.join().expect("worker thread"))
                .collect()
        });
        for result in results {
            assert_eq!(result, expected, "{metric:?} diverged under concurrency");
        }
    }
}
