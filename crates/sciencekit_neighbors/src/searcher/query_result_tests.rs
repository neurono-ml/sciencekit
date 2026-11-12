//! Companion tests for the shared query result types.

use ndarray::{Array2, array};

use super::query_result::{SKNeighborQuery, SKRadiusNeighbors};

#[test]
fn neighbor_query_exposes_row_major_distances_and_indices() {
    let distances = array![[0.0_f64, 1.5], [2.0, 3.5]];
    let indices = array![[4_usize, 1], [7, 2]];

    let query = SKNeighborQuery::new(distances, indices);

    assert_eq!(query.rows(), 2);
    assert_eq!(query.neighbors_count(), 2);
    assert_eq!(query.distances(), array![[0.0, 1.5], [2.0, 3.5]].view());
    assert_eq!(query.indices(), array![[4_usize, 1], [7, 2]].view());
}

#[test]
fn neighbor_query_supports_an_empty_batch() {
    let query = SKNeighborQuery::<f64>::new(Array2::zeros((0, 3)), Array2::zeros((0, 3)));

    assert_eq!(query.rows(), 0);
    assert_eq!(query.neighbors_count(), 3);
}

#[test]
fn neighbor_query_supports_zero_neighbors() {
    let query = SKNeighborQuery::<f64>::new(Array2::zeros((2, 0)), Array2::zeros((2, 0)));

    assert_eq!(query.rows(), 2);
    assert_eq!(query.neighbors_count(), 0);
}

#[test]
fn radius_neighbors_returns_per_row_slices() {
    let result = SKRadiusNeighbors::new(
        vec![vec![0.0_f64, 2.0], vec![]],
        vec![vec![0_usize, 3], vec![]],
    );

    assert_eq!(result.rows(), 2);
    assert_eq!(result.neighbors_count(0), 2);
    assert_eq!(result.distances(0), &[0.0, 2.0]);
    assert_eq!(result.indices(0), &[0, 3]);
    assert!(result.distances(1).is_empty());
    assert!(result.indices(1).is_empty());
    assert_eq!(result.neighbors_count(1), 0);
}
