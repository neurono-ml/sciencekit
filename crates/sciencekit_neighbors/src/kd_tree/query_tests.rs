//! Companion tests for exact KD-tree queries.

use ndarray::{Array2, array};
use sciencekit_common::SKDataView;

use super::SKKDTreeBuilder;
use crate::searcher::SKNearestNeighborsSearcher;
use crate::searcher::candidates::{SK_QUERY_PARALLEL_GRAIN, sk_should_run_parallel};

fn build(data: &Array2<f64>, leaf_size: usize) -> super::SKKDTree<f64> {
    SKKDTreeBuilder::<f64>::new()
        .leaf_size(leaf_size)
        .build(SKDataView::Dense(data.view()))
        .expect("valid training data")
}

fn assert_close(left: f64, right: f64) {
    assert!((left - right).abs() < 1e-12, "{left} != {right}");
}

#[test]
fn kd_query_returns_the_known_two_dimensional_neighbors() {
    let data = array![[0.0_f64, 0.0], [1.0, 0.0], [0.0, 1.0], [5.0, 5.0]];
    let tree = build(&data, 2);
    let queries = array![[0.1_f64, 0.05]];

    let result = tree
        .k_nearest(SKDataView::Dense(queries.view()), 2)
        .unwrap();

    assert_eq!(result.indices(), array![[0_usize, 1]].view());
    assert_close(result.distances()[[0, 0]], (0.1_f64 * 0.1 + 0.05 * 0.05).sqrt());
    assert_close(result.distances()[[0, 1]], (0.9_f64 * 0.9 + 0.05 * 0.05).sqrt());
}


#[test]
fn kd_query_finds_a_training_row_at_distance_zero_from_itself() {
    let data = array![[0.0_f64, 0.0], [1.0, 0.0], [3.0, 4.0], [9.0, 9.0]];
    let tree = build(&data, 2);
    let queries = data.row(2).to_owned().insert_axis(ndarray::Axis(0));

    let result = tree
        .k_nearest(SKDataView::Dense(queries.view()), 1)
        .unwrap();

    assert_eq!(result.indices()[[0, 0]], 2);
    assert_eq!(result.distances()[[0, 0]], 0.0);
}


#[test]
fn kd_query_supports_k_equal_to_one() {
    let data = array![[0.0_f64, 0.0], [1.0, 0.0], [5.0, 5.0]];
    let tree = build(&data, 1);
    let queries = array![[4.9_f64, 5.1]];

    let result = tree
        .k_nearest(SKDataView::Dense(queries.view()), 1)
        .unwrap();

    assert_eq!(result.indices()[[0, 0]], 2);
}


#[test]
fn kd_query_supports_k_equal_to_the_training_row_count() {
    let data = array![[0.0_f64, 0.0], [1.0, 0.0], [0.0, 1.0], [5.0, 5.0]];
    let tree = build(&data, 2);
    let queries = array![[0.0_f64, 0.0]];

    let result = tree
        .k_nearest(SKDataView::Dense(queries.view()), 4)
        .unwrap();

    let mut indices: Vec<usize> = result.indices().row(0).to_vec();
    indices.sort_unstable();
    assert_eq!(indices, vec![0, 1, 2, 3]);
    for column in 1..4 {
        assert!(result.distances()[[0, column - 1]] <= result.distances()[[0, column]]);
    }
}


#[test]
fn kd_query_returns_one_sorted_row_per_query_row() {
    let data = array![[0.0_f64, 0.0], [1.0, 0.0], [0.0, 1.0], [5.0, 5.0]];
    let tree = build(&data, 2);
    let queries = array![[0.0_f64, 0.0], [5.0, 5.0], [0.0, 1.0]];

    let result = tree
        .k_nearest(SKDataView::Dense(queries.view()), 3)
        .unwrap();

    assert_eq!(result.distances().shape(), &[3, 3]);
    assert_eq!(result.indices().shape(), &[3, 3]);
    for row in 0..3 {
        for column in 1..3 {
            assert!(result.distances()[[row, column - 1]] <= result.distances()[[row, column]]);
        }
    }
}

#[test]
fn kd_query_breaks_distance_ties_by_the_smallest_training_index() {
    let data = array![[1.0_f64, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]];
    let tree = build(&data, 2);
    let queries = array![[0.0_f64, 0.0]];

    let result = tree
        .k_nearest(SKDataView::Dense(queries.view()), 2)
        .unwrap();

    assert_eq!(result.indices(), array![[0_usize, 1]].view());
    assert_eq!(result.distances(), array![[1.0_f64, 1.0]].view());
}

#[test]
fn kd_query_tie_order_is_independent_of_the_tree_partition() {
    let data = array![[0.0_f64, 1.0], [1.0, 0.0], [0.0, -1.0], [-1.0, 0.0]];
    let tree = build(&data, 1);
    let queries = array![[0.0_f64, 0.0]];

    let result = tree
        .k_nearest(SKDataView::Dense(queries.view()), 1)
        .unwrap();

    assert_eq!(result.indices()[[0, 0]], 0);
}

#[test]
fn kd_radius_query_includes_the_boundary() {
    let data = array![[1.0_f64, 0.0], [2.0, 0.0]];
    let tree = build(&data, 1);
    let queries = array![[0.0_f64, 0.0]];

    let at_boundary = tree
        .radius_neighbors(SKDataView::Dense(queries.view()), 1.0)
        .unwrap();
    assert_eq!(at_boundary.indices(0), &[0]);
    assert_eq!(at_boundary.distances(0), &[1.0]);

    let both = tree
        .radius_neighbors(SKDataView::Dense(queries.view()), 2.0)
        .unwrap();
    assert_eq!(both.indices(0), &[0, 1]);
}

#[test]
fn kd_radius_query_represents_an_empty_neighborhood() {
    let data = array![[5.0_f64, 5.0]];
    let tree = build(&data, 1);
    let queries = array![[0.0_f64, 0.0]];

    let result = tree
        .radius_neighbors(SKDataView::Dense(queries.view()), 1.0)
        .unwrap();

    assert_eq!(result.rows(), 1);
    assert_eq!(result.neighbors_count(0), 0);
    assert!(result.indices(0).is_empty());
}

#[test]
fn kd_radius_query_sorts_distances_and_breaks_ties_by_index() {
    let data = array![[2.0_f64, 0.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]];
    let tree = build(&data, 2);
    let queries = array![[0.0_f64, 0.0]];

    let result = tree
        .radius_neighbors(SKDataView::Dense(queries.view()), 3.0)
        .unwrap();

    assert_eq!(result.indices(0), &[1, 2, 3, 0]);
    assert_eq!(result.distances(0), &[1.0, 1.0, 1.0, 2.0]);
}

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
