//! Companion tests for exact ball-tree queries.

use ndarray::{Array2, array};
use sciencekit_common::SKDataView;

use super::SKBallTreeBuilder;
use crate::metric::SKRequestedNeighborMetric;
use crate::searcher::SKNearestNeighborsSearcher;

fn build(data: &Array2<f64>, leaf_size: usize) -> super::SKBallTree<f64> {
    SKBallTreeBuilder::<f64>::new()
        .leaf_size(leaf_size)
        .build(SKDataView::Dense(data.view()))
        .expect("valid training data")
}

#[test]
fn ball_query_returns_the_known_neighbors() {
    let data = array![[0.0_f64, 0.0], [1.0, 0.0], [0.0, 1.0], [5.0, 5.0]];
    let tree = build(&data, 2);
    let queries = array![[0.1_f64, 0.05]];

    let result = tree
        .k_nearest(SKDataView::Dense(queries.view()), 2)
        .unwrap();

    assert_eq!(result.indices(), array![[0_usize, 1]].view());
    assert!((result.distances()[[0, 0]] - 0.11180339887498948).abs() < 1e-12);
}

#[test]
fn ball_query_supports_k_equal_to_the_training_row_count() {
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
fn ball_query_returns_one_sorted_row_per_query_row() {
    let data = array![[0.0_f64, 0.0], [1.0, 0.0], [0.0, 1.0], [5.0, 5.0]];
    let tree = build(&data, 2);
    let queries = array![[0.0_f64, 0.0], [5.0, 5.0]];

    let result = tree
        .k_nearest(SKDataView::Dense(queries.view()), 3)
        .unwrap();

    assert_eq!(result.distances().shape(), &[2, 3]);
    assert_eq!(result.indices().shape(), &[2, 3]);
    for row in 0..2 {
        for column in 1..3 {
            assert!(result.distances()[[row, column - 1]] <= result.distances()[[row, column]]);
        }
    }
}

#[test]
fn ball_query_breaks_distance_ties_by_the_smallest_training_index() {
    let data = array![[1.0_f64, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]];
    let tree = build(&data, 2);
    let queries = array![[0.0_f64, 0.0]];

    let result = tree
        .k_nearest(SKDataView::Dense(queries.view()), 2)
        .unwrap();

    assert_eq!(result.indices(), array![[0_usize, 1]].view());
}

#[test]
fn ball_radius_query_includes_the_boundary() {
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
fn ball_radius_query_sorts_and_represents_empty_neighborhoods() {
    let data = array![[2.0_f64, 0.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]];
    let tree = build(&data, 2);
    let queries = array![[0.0_f64, 0.0], [20.0, 20.0]];

    let result = tree
        .radius_neighbors(SKDataView::Dense(queries.view()), 3.0)
        .unwrap();

    assert_eq!(result.indices(0), &[1, 2, 3, 0]);
    assert_eq!(result.distances(0), &[1.0, 1.0, 1.0, 2.0]);
    assert!(result.indices(1).is_empty());
}

#[test]
fn ball_query_parallel_path_equals_the_sequential_path_on_a_large_batch() {
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

    let sequential_radius = tree.radius_neighbors_with_mode(queries.view(), 1.5, false);
    let parallel_radius = tree.radius_neighbors_with_mode(queries.view(), 1.5, true);
    assert_eq!(sequential_radius, parallel_radius);
}

/// Exact-boundary radii must be inclusive for every metric, including when the
/// whole index is one leaf.
#[test]
fn ball_radius_query_includes_the_exact_boundary_for_every_metric() {
    let training =
        ndarray::Array2::from_shape_vec((5, 1), vec![2.0_f64, 2.0, 0.0, 0.0, 0.0]).unwrap();
    let query = ndarray::Array2::from_shape_vec((1, 1), vec![3.0_f64]).unwrap();
    for metric in [
        SKRequestedNeighborMetric::Euclidean,
        SKRequestedNeighborMetric::SquaredEuclidean,
        SKRequestedNeighborMetric::Manhattan,
        SKRequestedNeighborMetric::Chebyshev,
    ] {
        let tree = SKBallTreeBuilder::<f64>::new()
            .metric(metric)
            .build(SKDataView::Dense(training.view()))
            .unwrap();
        let radius = match metric {
            SKRequestedNeighborMetric::SquaredEuclidean => 1.0,
            _ => 1.0,
        };
        let neighbors = tree
            .radius_neighbors(
                sciencekit_common::SKDataView::try_from(&query).unwrap(),
                radius,
            )
            .unwrap();
        assert_eq!(
            neighbors.neighbors_count(0),
            2,
            "{metric:?} must include the rows exactly at the radius"
        );
    }
}

/// Equidistant candidates keep the smallest training index, including when the
/// node holding the smaller index has a bound exactly equal to the worst.
#[test]
fn ball_query_keeps_deterministic_ties_at_the_pruning_boundary() {
    let training =
        ndarray::Array2::from_shape_vec((4, 2), vec![0.0_f64, 2.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            .unwrap();
    let query = ndarray::Array2::from_shape_vec((1, 2), vec![1.0_f64, 1.0]).unwrap();
    let tree = SKBallTreeBuilder::<f64>::new()
        .metric(SKRequestedNeighborMetric::SquaredEuclidean)
        .leaf_size(1)
        .build(SKDataView::Dense(training.view()))
        .unwrap();
    let neighbors = tree
        .k_nearest(sciencekit_common::SKDataView::try_from(&query).unwrap(), 2)
        .unwrap();
    assert_eq!(neighbors.indices()[[0, 0]], 0);
    assert_eq!(neighbors.indices()[[0, 1]], 1);
}

/// The same logical data partitioned by a different leaf size answers
/// identically, ties included.
#[test]
fn ball_partition_does_not_change_the_answer() {
    let training = ndarray::Array2::from_shape_vec(
        (8, 2),
        vec![
            0.0_f64, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 0.0, 0.0, 2.0, 3.0, 0.0, 0.0, 3.0,
        ],
    )
    .unwrap();
    let query = ndarray::Array2::from_shape_vec((2, 2), vec![0.5_f64, 0.5, 2.0, 2.0]).unwrap();
    let mut answers = Vec::new();
    for leaf_size in [1_usize, 2, 40] {
        let tree = SKBallTreeBuilder::<f64>::new()
            .leaf_size(leaf_size)
            .build(SKDataView::Dense(training.view()))
            .unwrap();
        answers.push(
            tree.k_nearest(sciencekit_common::SKDataView::try_from(&query).unwrap(), 3)
                .unwrap(),
        );
    }
    for answer in &answers[1..] {
        assert_eq!(answer.indices(), answers[0].indices());
        assert_eq!(answer.distances(), answers[0].distances());
    }
}
