//! Acceptance tests on large data for both indexes.

use ndarray::{Array2, Axis, Slice};
use sciencekit_common::SKDataView;

use crate::ball_tree::SKBallTreeBuilder;
use crate::kd_tree::SKKDTreeBuilder;
use crate::searcher::SKNearestNeighborsSearcher;

fn large_training() -> Array2<f64> {
    Array2::from_shape_fn((20_000, 2), |(row, column)| {
        ((row * 37 + column * 11) % 1009) as f64 / 17.0
    })
}

fn large_queries() -> Array2<f64> {
    Array2::from_shape_fn((10_000, 2), |(row, column)| {
        ((row * 53 + column * 19) % 1013) as f64 / 13.0
    })
}

#[test]
fn acceptance_large_batches_agree_between_both_trees() {
    let data = large_training();
    let queries = large_queries();

    let kd = SKKDTreeBuilder::<f64>::new()
        .leaf_size(32)
        .build(SKDataView::Dense(data.view()))
        .unwrap();
    let ball = SKBallTreeBuilder::<f64>::new()
        .leaf_size(32)
        .build(SKDataView::Dense(data.view()))
        .unwrap();

    let kd_result = kd
        .k_nearest(SKDataView::Dense(queries.view()), 5)
        .unwrap();
    let ball_result = ball
        .k_nearest(SKDataView::Dense(queries.view()), 5)
        .unwrap();

    assert_eq!(kd_result.distances().shape(), &[10_000, 5]);
    assert_eq!(ball_result.distances().shape(), &[10_000, 5]);
    assert_eq!(kd_result.indices(), ball_result.indices());
    assert_eq!(kd_result.distances(), ball_result.distances());

    for row in 0..queries.nrows() {
        for column in 1..5 {
            assert!(
                kd_result.distances()[[row, column - 1]] <= kd_result.distances()[[row, column]]
            );
        }
    }
}

#[test]
fn acceptance_large_radius_batches_agree_between_both_trees() {
    let data = large_training();
    let queries = large_queries()
        .slice_axis(Axis(0), Slice::from(0..1_000))
        .to_owned();

    let kd = SKKDTreeBuilder::<f64>::new()
        .leaf_size(32)
        .build(SKDataView::Dense(data.view()))
        .unwrap();
    let ball = SKBallTreeBuilder::<f64>::new()
        .leaf_size(32)
        .build(SKDataView::Dense(data.view()))
        .unwrap();

    let kd_result = kd
        .radius_neighbors(SKDataView::Dense(queries.view()), 0.75)
        .unwrap();
    let ball_result = ball
        .radius_neighbors(SKDataView::Dense(queries.view()), 0.75)
        .unwrap();

    assert_eq!(kd_result.rows(), 1_000);
    assert_eq!(kd_result, ball_result);
}
