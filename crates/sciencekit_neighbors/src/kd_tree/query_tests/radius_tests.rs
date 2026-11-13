//! Inclusive KD-tree radius query tests.

use ndarray::array;
use sciencekit_common::SKDataView;

use super::fixtures::build;
use crate::searcher::SKNearestNeighborsSearcher;

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
