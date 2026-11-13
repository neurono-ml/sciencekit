//! Companion tests for ball-tree construction.

use ndarray::{Array2, array};
use sciencekit_common::{SKDataView, SKError};

use super::SKBallTreeBuilder;

fn grid(rows: usize) -> Array2<f64> {
    Array2::from_shape_fn((rows, 3), |(row, column)| {
        ((row * 7 + column * 3) % 43) as f64 / 3.0
    })
}

fn euclidean(data: &Array2<f64>, row: usize, centroid: &[f64]) -> f64 {
    data.row(row)
        .iter()
        .zip(centroid.iter())
        .map(|(&value, &other)| (value - other) * (value - other))
        .sum::<f64>()
        .sqrt()
}

#[test]
fn ball_build_rejects_an_empty_training_set() {
    let empty = Array2::<f64>::zeros((0, 3));
    let error = SKBallTreeBuilder::<f64>::new()
        .build(SKDataView::Dense(empty.view()))
        .unwrap_err();

    match error {
        SKError::EmptyInput { operation } => assert_eq!(operation, "build"),
        other => panic!("expected EmptyInput, found {other:?}"),
    }
}

#[test]
fn ball_build_keeps_every_member_inside_its_node_radius() {
    let data = grid(120);
    let tree = SKBallTreeBuilder::<f64>::new()
        .leaf_size(5)
        .build(SKDataView::Dense(data.view()))
        .unwrap();

    for node in tree.nodes() {
        for &row in &tree.permutation()[node.start()..node.end()] {
            assert!(
                euclidean(&data, row, node.centroid()) <= node.radius() + 1e-12,
                "row {row} lies outside its node radius"
            );
        }
    }
}

#[test]
fn ball_build_produces_contiguous_leaf_size_bounded_leaves() {
    let data = grid(101);
    let tree = SKBallTreeBuilder::<f64>::new()
        .leaf_size(8)
        .build(SKDataView::Dense(data.view()))
        .unwrap();

    let mut permutation = tree.permutation().to_vec();
    permutation.sort_unstable();
    assert_eq!(permutation, (0..101).collect::<Vec<_>>());

    let mut leaves: Vec<(usize, usize)> = tree
        .nodes()
        .iter()
        .filter(|node| node.is_leaf())
        .map(|node| (node.start(), node.end()))
        .collect();
    leaves.sort_unstable();
    for window in leaves.windows(2) {
        assert_eq!(window[0].1, window[1].0);
    }
    for (start, end) in leaves {
        assert!(end > start);
        assert!(end - start <= 8);
    }
}

#[test]
fn ball_build_handles_a_single_row() {
    let data = array![[1.0_f64, 2.0, 3.0]];
    let tree = SKBallTreeBuilder::<f64>::new()
        .build(SKDataView::Dense(data.view()))
        .unwrap();

    assert_eq!(tree.node_count(), 1);
    assert_eq!(tree.depth(), 1);
    assert_eq!(tree.nodes()[0].radius(), 0.0);
    assert_eq!(tree.nodes()[0].centroid(), &[1.0, 2.0, 3.0]);
}

#[test]
fn ball_build_is_deterministic_for_duplicated_points() {
    let data = array![
        [1.0_f64, 1.0, 1.0],
        [1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0],
        [0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0]
    ];

    let first = SKBallTreeBuilder::<f64>::new()
        .leaf_size(2)
        .build(SKDataView::Dense(data.view()))
        .unwrap();
    let second = SKBallTreeBuilder::<f64>::new()
        .leaf_size(2)
        .build(SKDataView::Dense(data.view()))
        .unwrap();

    assert_eq!(first.permutation(), second.permutation());
    assert_eq!(first.node_count(), second.node_count());
    for (left, right) in first.nodes().iter().zip(second.nodes()) {
        assert_eq!(left.start(), right.start());
        assert_eq!(left.end(), right.end());
        assert_eq!(left.centroid(), right.centroid());
        assert_eq!(left.radius(), right.radius());
    }
}
