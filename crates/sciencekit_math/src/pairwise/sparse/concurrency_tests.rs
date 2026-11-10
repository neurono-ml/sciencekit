//! TDD tests for concurrent sparse distance computation.

use ndarray::Array2;
use sprs::CsMat;

use super::fixtures::large_sparse_fixture;
use super::{
    sk_sparse_chebyshev_distance_matrix, sk_sparse_cosine_distance_matrix,
    sk_sparse_euclidean_distance_matrix, sk_sparse_manhattan_distance_matrix,
    sk_sparse_squared_euclidean_distance_matrix,
};

/// A metric kernel over two borrowed CSR views.
type SparseKernel = fn(&CsMat<f64>, &CsMat<f64>) -> Array2<f64>;

/// Several threads computing the same sparse distance matrix simultaneously
/// all reproduce the single-threaded result.
#[test]
fn sparse_pairwise_concurrent_calls_are_deterministic() {
    let sparse = large_sparse_fixture(80, 48);
    let kernels: [(&str, SparseKernel); 5] = [
        ("squared_euclidean", |left, right| {
            sk_sparse_squared_euclidean_distance_matrix(&left.view(), &right.view())
        }),
        ("euclidean", |left, right| {
            sk_sparse_euclidean_distance_matrix(&left.view(), &right.view())
        }),
        ("manhattan", |left, right| {
            sk_sparse_manhattan_distance_matrix(&left.view(), &right.view())
        }),
        ("chebyshev", |left, right| {
            sk_sparse_chebyshev_distance_matrix(&left.view(), &right.view())
        }),
        ("cosine", |left, right| {
            sk_sparse_cosine_distance_matrix(&left.view(), &right.view())
        }),
    ];
    for (name, compute) in kernels {
        let expected = compute(&sparse, &sparse);
        let concurrent: Vec<Array2<f64>> = std::thread::scope(|scope| {
            let handles: Vec<_> = (0..8)
                .map(|_| scope.spawn(|| compute(&sparse, &sparse)))
                .collect();
            handles
                .into_iter()
                .map(|handle| handle.join().expect("worker thread"))
                .collect()
        });
        for result in concurrent {
            assert_eq!(result, expected, "{name} diverged under concurrency");
        }
    }
}
