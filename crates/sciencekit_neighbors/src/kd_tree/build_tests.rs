//! Companion tests for KD-tree construction.

use ndarray::{Array2, array};
use sciencekit_common::{SKBackendKind, SKDataView, SKError, SKExecutionMode};

use super::SKKDTreeBuilder;
use crate::metric::SKNeighborMetric;

fn grid(rows: usize) -> Array2<f64> {
    Array2::from_shape_fn((rows, 2), |(row, column)| {
        if column == 0 {
            row as f64
        } else {
            (row * row) as f64
        }
    })
}

#[test]
fn kd_build_rejects_an_empty_training_set() {
    let empty = Array2::<f64>::zeros((0, 2));
    let error = SKKDTreeBuilder::<f64>::new()
        .build(SKDataView::Dense(empty.view()))
        .unwrap_err();

    match error {
        SKError::EmptyInput { operation } => assert_eq!(operation, "build"),
        other => panic!("expected EmptyInput, found {other:?}"),
    }
}


#[test]
fn kd_build_rejects_zero_leaf_size() {
    let data = grid(10);
    let error = SKKDTreeBuilder::<f64>::new()
        .leaf_size(0)
        .build(SKDataView::Dense(data.view()))
        .unwrap_err();

    match error {
        SKError::InvalidHyperparameter { name, .. } => assert_eq!(name, "leaf_size"),
        other => panic!("expected InvalidHyperparameter, found {other:?}"),
    }
}


#[test]
fn kd_build_handles_a_single_row() {
    let data = array![[1.0_f64, 2.0]];
    let tree = SKKDTreeBuilder::<f64>::new()
        .build(SKDataView::Dense(data.view()))
        .unwrap();

    assert_eq!(tree.training_rows(), 1);
    assert_eq!(tree.node_count(), 1);
    assert_eq!(tree.depth(), 1);
    assert!(tree.nodes()[0].is_leaf());
}


#[test]
fn kd_build_produces_a_permutation_and_leaf_size_bounded_contiguous_leaves() {
    let data = grid(101);
    let tree = SKKDTreeBuilder::<f64>::new()
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
    assert_eq!(leaves.first().map(|range| range.0), Some(0));
    assert_eq!(leaves.last().map(|range| range.1), Some(101));
    for window in leaves.windows(2) {
        assert_eq!(window[0].1, window[1].0, "leaves must be contiguous");
    }
    for (start, end) in leaves {
        assert!(end > start);
        assert!(end - start <= 8);
    }
}


#[test]
fn kd_build_encloses_every_member_inside_the_node_box() {
    let data = grid(50);
    let tree = SKKDTreeBuilder::<f64>::new()
        .leaf_size(4)
        .build(SKDataView::Dense(data.view()))
        .unwrap();

    for node in tree.nodes() {
        for &row in &tree.permutation()[node.start()..node.end()] {
            for (column, &value) in data.row(row).iter().enumerate() {
                assert!(value >= node.minimum()[column]);
                assert!(value <= node.maximum()[column]);
            }
        }
    }
}

