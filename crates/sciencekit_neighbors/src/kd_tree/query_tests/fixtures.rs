//! Shared fixtures for the KD-tree query tests.

use ndarray::Array2;
use sciencekit_common::SKDataView;

use super::super::SKKDTree;
use super::super::SKKDTreeBuilder;

pub(super) fn build(data: &Array2<f64>, leaf_size: usize) -> SKKDTree<f64> {
    SKKDTreeBuilder::<f64>::new()
        .leaf_size(leaf_size)
        .build(SKDataView::Dense(data.view()))
        .expect("valid training data")
}

pub(super) fn assert_close(left: f64, right: f64) {
    assert!((left - right).abs() < 1e-12, "{left} != {right}");
}
