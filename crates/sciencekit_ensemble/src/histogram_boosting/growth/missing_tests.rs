//! Missing-value routing tests (task 3.2).

use ndarray::Array2;
use sciencekit_common::SKDataView;
use sciencekit_tree::SKMaximumFeatures;

use crate::histogram_boosting::binning::SKBinnedDataset;
use crate::histogram_boosting::histogram::SKHistogramSplit;

use super::leaf_wise::{SKHistogramGrowthParameters, sk_grow_histogram_tree};

fn parameters() -> SKHistogramGrowthParameters<f64> {
    SKHistogramGrowthParameters {
        maximum_leaf_nodes: 2,
        minimum_samples_leaf: 1,
        l2_regularization: 0.0,
        maximum_features: SKMaximumFeatures::All,
        random_seed: 0,
    }
}

fn grow(values: Vec<f64>, gradients: &[f64]) -> super::tree::SKHistogramTree<f64> {
    let rows = values.len();
    let dense = Array2::from_shape_vec((rows, 1), values).expect("fixture");
    let dataset = SKBinnedDataset::build(&SKDataView::Dense(dense.view()), &[], 8).expect("build");
    let hessians = vec![1.0; rows];
    let row_indices: Vec<usize> = (0..rows).collect();
    sk_grow_histogram_tree::<f64>(&dataset, gradients, &hessians, &row_indices, &parameters()).tree
}

fn root_default_left(tree: &super::tree::SKHistogramTree<f64>) -> bool {
    match tree.nodes()[0].split().expect("root split") {
        SKHistogramSplit::Numeric { default_left, .. } => *default_left,
        _ => panic!("expected a numeric split"),
    }
}

#[test]
fn hist_missing_learns_the_left_direction() {
    let tree = grow(vec![0.0, 1.0, f64::NAN, 3.0], &[5.0, 5.0, 5.0, 0.1]);
    assert!(root_default_left(&tree));
    let missing_leaf = tree.leaf_value_for(|feature| {
        assert_eq!(feature, 0);
        tree_bin_missing()
    });
    let left_leaf = tree.leaf_value_for(|_| 0);
    assert!((missing_leaf - left_leaf).abs() < 1e-12);
}

fn tree_bin_missing() -> u8 {
    8
}

#[test]
fn hist_missing_learns_the_right_direction() {
    let tree = grow(vec![0.0, 1.0, f64::NAN, 3.0], &[1.0, 5.0, 9.0, 9.0]);
    assert!(!root_default_left(&tree));
    let missing_leaf = tree.leaf_value_for(|_| 8);
    let right_leaf = tree.leaf_value_for(|_| 7);
    assert!((missing_leaf - right_leaf).abs() < 1e-12);
}

#[test]
fn hist_missing_without_missing_rows_uses_deterministic_direction() {
    let tree = grow(vec![0.0, 1.0, 2.0, 3.0], &[1.0, 1.0, -1.0, -1.0]);
    assert!(!root_default_left(&tree));
}
