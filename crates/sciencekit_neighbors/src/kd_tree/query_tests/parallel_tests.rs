//! KD-tree row-parallel query tests.

use ndarray::Array2;

use super::fixtures::build;
use crate::searcher::candidates::{SK_QUERY_PARALLEL_GRAIN, sk_should_run_parallel};

#[test]
fn kd_query_parallel_path_equals_the_sequential_path_on_a_large_batch() {
    let data = Array2::from_shape_fn((3000, 3), |(row, column)| {
        ((row * 13 + column * 7) % 97) as f64 / 7.0
    });
    let queries = Array2::from_shape_fn((200, 3), |(row, column)| {
        ((row * 5 + column * 11) % 89) as f64 / 5.0
    });
    let tree = build(&data, 16);

    let sequential = tree.k_nearest_with_mode(queries.view(), 5, false);
    let parallel = tree.k_nearest_with_mode(queries.view(), 5, true);

    assert_eq!(sequential.distances(), parallel.distances());
    assert_eq!(sequential.indices(), parallel.indices());
}

#[test]
fn kd_query_parallel_grain_boundary_is_explicit() {
    assert!(!sk_should_run_parallel(0, SK_QUERY_PARALLEL_GRAIN));
    assert!(!sk_should_run_parallel(1, SK_QUERY_PARALLEL_GRAIN - 1));
    assert!(sk_should_run_parallel(1, SK_QUERY_PARALLEL_GRAIN));
    assert!(sk_should_run_parallel(2, SK_QUERY_PARALLEL_GRAIN));
}
