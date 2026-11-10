//! Deterministic KD-tree tie-ordering tests.

use ndarray::array;
use sciencekit_common::SKDataView;

use super::fixtures::build;
use crate::searcher::SKNearestNeighborsSearcher;

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
