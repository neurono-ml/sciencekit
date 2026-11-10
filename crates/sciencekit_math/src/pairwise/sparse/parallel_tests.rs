//! TDD tests for the row-parallel sparse paths.

use ndarray::Array2;
use sprs::CsMat;

use super::fixtures::large_sparse_fixture;
use super::parallel::SK_SPARSE_PAIRWISE_PARALLEL_GRAIN;
use super::{
    sk_sparse_chebyshev_distance_matrix, sk_sparse_cosine_distance_matrix,
    sk_sparse_euclidean_distance_matrix, sk_sparse_manhattan_distance_matrix,
    sk_sparse_squared_euclidean_distance_matrix,
};

/// A metric kernel over two borrowed CSR views.
type SparseKernel = fn(&CsMat<f64>, &CsMat<f64>) -> Array2<f64>;

/// Run a computation through a single-thread rayon pool, i.e. sequentially.
fn run_sequentially(compute: impl FnOnce() -> Array2<f64> + Send) -> Array2<f64> {
    rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .expect("single-thread pool")
        .install(compute)
}

/// A large sparse input whose pair count crosses the parallel grain.
fn large_fixture() -> CsMat<f64> {
    // 160 rows × 160 columns of pairs, well above the parallel grain.
    let sparse = large_sparse_fixture(160, 64);
    let _pair_count_is_above_grain = SK_SPARSE_PAIRWISE_PARALLEL_GRAIN < 160 * 160;
    sparse
}

/// Assert one metric's parallel result equals its sequential result.
fn assert_parallel_matches(name: &str, compute: SparseKernel, sparse: &CsMat<f64>) {
    let sequential = run_sequentially(|| compute(sparse, sparse));
    let parallel = compute(sparse, sparse);
    assert_eq!(parallel.shape(), sequential.shape());
    for ((index, parallel_value), sequential_value) in
        parallel.indexed_iter().zip(sequential.iter())
    {
        assert!(
            (parallel_value - sequential_value).abs() < 1e-12,
            "{name} mismatch at {index:?}: {parallel_value} vs {sequential_value}"
        );
    }
}

/// The parallel path equals the sequential path for every sparse metric.
#[test]
fn sparse_pairwise_parallel_equals_sequential() {
    let sparse = large_fixture();
    let cases: [(&str, SparseKernel); 5] = [
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
    for (name, compute) in cases {
        assert_parallel_matches(name, compute, &sparse);
    }
}
