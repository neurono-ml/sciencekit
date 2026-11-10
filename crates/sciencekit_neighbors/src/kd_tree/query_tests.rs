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
