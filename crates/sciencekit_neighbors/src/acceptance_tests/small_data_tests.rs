//! Acceptance tests on tiny data, every metric and both indexes.

use ndarray::{Array1, Array2, array};
use sciencekit_common::SKDataView;

use crate::ball_tree::SKBallTreeBuilder;
use crate::kd_tree::SKKDTreeBuilder;
use crate::metric::SKNeighborMetric;
use crate::searcher::SKNearestNeighborsSearcher;

fn dataset() -> Array2<f64> {
    array![
        [0.0_f64, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [3.0, 4.0],
        [-2.0, 1.0],
        [0.5, -0.5]
    ]
}

fn brute_force(
    data: &Array2<f64>,
    query: Array1<f64>,
    neighbors_count: usize,
    metric: SKNeighborMetric,
) -> Vec<(f64, usize)> {
    let mut all: Vec<(f64, usize)> = (0..data.nrows())
        .map(|row| (metric.point_distance(query.view(), data.row(row)), row))
        .collect();
    all.sort_by(|left, right| {
        left.0
            .partial_cmp(&right.0)
            .unwrap()
            .then_with(|| left.1.cmp(&right.1))
    });
    all.truncate(neighbors_count);
    all
}

#[test]
fn acceptance_tiny_one_row_one_query_on_both_trees() {
    let data = array![[2.0_f64, -3.0]];
    let queries = array![[2.0_f64, -3.0]];

    let kd = SKKDTreeBuilder::<f64>::new()
        .build(SKDataView::Dense(data.view()))
        .unwrap();
    let ball = SKBallTreeBuilder::<f64>::new()
        .build(SKDataView::Dense(data.view()))
        .unwrap();

    for result in [
        kd.k_nearest(SKDataView::Dense(queries.view()), 1).unwrap(),
        ball.k_nearest(SKDataView::Dense(queries.view()), 1)
            .unwrap(),
    ] {
        assert_eq!(result.indices()[[0, 0]], 0);
        assert_eq!(result.distances()[[0, 0]], 0.0);
    }
}

#[test]
fn acceptance_every_metric_matches_brute_force_on_both_trees() {
    let data = dataset();
    let queries = array![[0.1_f64, 0.2], [2.5, 3.5], [-1.0, 0.0]];

    for metric in SKNeighborMetric::ALL {
        let kd = SKKDTreeBuilder::<f64>::new()
            .leaf_size(2)
            .metric(metric)
            .build(SKDataView::Dense(data.view()))
            .unwrap();
        let ball = SKBallTreeBuilder::<f64>::new()
            .leaf_size(2)
            .metric(metric)
            .build(SKDataView::Dense(data.view()))
            .unwrap();

        let kd_result = kd.k_nearest(SKDataView::Dense(queries.view()), 3).unwrap();
        let ball_result = ball
            .k_nearest(SKDataView::Dense(queries.view()), 3)
            .unwrap();

        for row in 0..queries.nrows() {
            let expected = brute_force(&data, queries.row(row).to_owned(), 3, metric);
            for (column, (distance, index)) in expected.into_iter().enumerate() {
                assert_eq!(kd_result.indices()[[row, column]], index, "{metric:?}");
                assert_eq!(ball_result.indices()[[row, column]], index, "{metric:?}");
                assert!((kd_result.distances()[[row, column]] - distance).abs() < 1e-12);
                assert!((ball_result.distances()[[row, column]] - distance).abs() < 1e-12);
            }
        }
    }
}

#[test]
fn acceptance_radius_queries_match_brute_force_on_both_trees() {
    let data = dataset();
    let queries = array![[0.0_f64, 0.0], [3.0, 4.0]];

    for metric in SKNeighborMetric::ALL {
        let kd = SKKDTreeBuilder::<f64>::new()
            .metric(metric)
            .build(SKDataView::Dense(data.view()))
            .unwrap();
        let ball = SKBallTreeBuilder::<f64>::new()
            .metric(metric)
            .build(SKDataView::Dense(data.view()))
            .unwrap();

        for (row, query) in queries.rows().into_iter().enumerate() {
            let expected: Vec<usize> = (0..data.nrows())
                .filter(|&index| metric.point_distance(query, data.row(index)) <= 2.5)
                .collect();
            let kd_result = kd
                .radius_neighbors(SKDataView::Dense(queries.view()), 2.5)
                .unwrap();
            let ball_result = ball
                .radius_neighbors(SKDataView::Dense(queries.view()), 2.5)
                .unwrap();
            let mut kd_indices = kd_result.indices(row).to_vec();
            let mut ball_indices = ball_result.indices(row).to_vec();
            kd_indices.sort_unstable();
            ball_indices.sort_unstable();
            let mut expected = expected;
            expected.sort_unstable();
            assert_eq!(kd_indices, expected, "{metric:?}");
            assert_eq!(ball_indices, expected, "{metric:?}");
        }
    }
}

#[test]
fn acceptance_deterministic_ties_hold_on_both_trees() {
    let data = array![[1.0_f64, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]];
    let queries = array![[0.0_f64, 0.0]];

    let kd = SKKDTreeBuilder::<f64>::new()
        .leaf_size(1)
        .build(SKDataView::Dense(data.view()))
        .unwrap();
    let ball = SKBallTreeBuilder::<f64>::new()
        .leaf_size(1)
        .build(SKDataView::Dense(data.view()))
        .unwrap();

    for result in [
        kd.k_nearest(SKDataView::Dense(queries.view()), 3).unwrap(),
        ball.k_nearest(SKDataView::Dense(queries.view()), 3)
            .unwrap(),
    ] {
        assert_eq!(result.indices(), array![[0_usize, 1, 2]].view());
    }
}
