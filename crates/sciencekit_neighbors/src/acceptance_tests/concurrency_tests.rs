//! Acceptance tests for thread safety and concurrent query batches.

use ndarray::{Array2, Axis, Slice};
use sciencekit_common::SKDataView;

use crate::ball_tree::{SKBallTree, SKBallTreeBuilder};
use crate::kd_tree::{SKKDTree, SKKDTreeBuilder};
use crate::searcher::SKNearestNeighborsSearcher;

fn assert_send_sync<T: Send + Sync>() {}

#[test]
fn acceptance_both_index_types_are_send_and_sync() {
    assert_send_sync::<SKKDTree<f64>>();
    assert_send_sync::<SKBallTree<f64>>();
}

#[test]
fn acceptance_one_shared_index_serves_concurrent_batches() {
    let data = Array2::from_shape_fn((4_000, 2), |(row, column)| {
        ((row * 29 + column * 7) % 211) as f64
    });
    let queries = Array2::from_shape_fn((800, 2), |(row, column)| {
        ((row * 17 + column * 13) % 223) as f64
    });

    let kd = SKKDTreeBuilder::<f64>::new()
        .leaf_size(16)
        .build(SKDataView::Dense(data.view()))
        .unwrap();
    let ball = SKBallTreeBuilder::<f64>::new()
        .leaf_size(16)
        .build(SKDataView::Dense(data.view()))
        .unwrap();

    let expected_kd = kd.k_nearest(SKDataView::Dense(queries.view()), 4).unwrap();
    let expected_ball = ball
        .k_nearest(SKDataView::Dense(queries.view()), 4)
        .unwrap();

    std::thread::scope(|scope| {
        let mut handles = Vec::new();
        for chunk in 0..4 {
            let batch = queries
                .slice_axis(Axis(0), Slice::from(chunk * 200..(chunk + 1) * 200))
                .to_owned();
            let kd = &kd;
            let ball = &ball;
            handles.push(scope.spawn(move || {
                let kd_result = kd.k_nearest(SKDataView::Dense(batch.view()), 4).unwrap();
                let ball_result = ball.k_nearest(SKDataView::Dense(batch.view()), 4).unwrap();
                (chunk, kd_result, ball_result)
            }));
        }

        for handle in handles {
            let (chunk, kd_result, ball_result) = handle.join().expect("thread joins");
            let range = Slice::from(chunk * 200..(chunk + 1) * 200);
            assert_eq!(
                kd_result.indices(),
                expected_kd.indices().slice_axis(Axis(0), range)
            );
            assert_eq!(
                kd_result.distances(),
                expected_kd.distances().slice_axis(Axis(0), range)
            );
            assert_eq!(
                ball_result.indices(),
                expected_ball.indices().slice_axis(Axis(0), range)
            );
            assert_eq!(
                ball_result.distances(),
                expected_ball.distances().slice_axis(Axis(0), range)
            );
        }
    });
}
