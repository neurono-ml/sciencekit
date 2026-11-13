//! Leaf-wise histogram growth tests (task 3.1).

use ndarray::Array2;
use sciencekit_common::SKDataView;
use sciencekit_tree::SKMaximumFeatures;

use crate::histogram_boosting::binning::SKBinnedDataset;
use crate::histogram_boosting::histogram::SKHistogramSplit;

use super::leaf_wise::{SKHistogramGrowthParameters, sk_grow_histogram_tree};
use super::tree::{SKHistogramNode, SKHistogramTree};

fn parameters(maximum_leaf_nodes: usize) -> SKHistogramGrowthParameters<f64> {
    SKHistogramGrowthParameters {
        maximum_leaf_nodes,
        minimum_samples_leaf: 1,
        l2_regularization: 0.0,
        maximum_features: SKMaximumFeatures::All,
        random_seed: 0,
    }
}

fn count_leaves(tree: &SKHistogramTree<f64>) -> usize {
    tree.nodes()
        .iter()
        .filter(|node| node.left_child().is_none())
        .count()
}

#[test]
fn hist_growth_splits_the_highest_gain_leaf() {
    let dense = Array2::from_shape_vec((4, 1), vec![0.0, 1.0, 2.0, 3.0]).expect("fixture");
    let dataset = SKBinnedDataset::build(&SKDataView::Dense(dense.view()), &[], 4).expect("build");
    let gradients = [1.0, 1.0, -1.0, -1.0];
    let hessians = [1.0; 4];
    let rows: Vec<usize> = (0..4).collect();
    let result =
        sk_grow_histogram_tree::<f64>(&dataset, &gradients, &hessians, &rows, &parameters(3));
    assert_eq!(result.tree.node_count(), 3);
    assert_eq!(count_leaves(&result.tree), 2);
    assert!((result.feature_importances[0] - 1.0).abs() < 1e-12);
    let root = &result.tree.nodes()[0];
    match root.split().expect("root split") {
        SKHistogramSplit::Numeric { maximum_bin, .. } => assert_eq!(*maximum_bin, 1),
        _ => panic!("expected a numeric split"),
    }
    let left_value = result
        .tree
        .leaf_value_for(|feature| dataset.bin(feature, 0));
    let right_value = result
        .tree
        .leaf_value_for(|feature| dataset.bin(feature, 3));
    assert!((left_value - 1.0).abs() < 1e-12);
    assert!((right_value + 1.0).abs() < 1e-12);
}

#[test]
fn hist_growth_respects_the_maximum_leaf_budget() {
    let values: Vec<f64> = (0..16).map(|value| value as f64).collect();
    let dense = Array2::from_shape_vec((16, 1), values).expect("fixture");
    let dataset = SKBinnedDataset::build(&SKDataView::Dense(dense.view()), &[], 16).expect("build");
    let gradients: Vec<f64> = (0..16)
        .map(|row| if row % 2 == 0 { 1.0 } else { -1.0 })
        .collect();
    let hessians = vec![1.0; 16];
    let rows: Vec<usize> = (0..16).collect();
    let result =
        sk_grow_histogram_tree::<f64>(&dataset, &gradients, &hessians, &rows, &parameters(3));
    assert!(count_leaves(&result.tree) <= 3);
    assert_eq!(result.tree.node_count(), 5);
}

#[test]
fn hist_growth_creates_a_leaf_node_without_a_split() {
    let leaf: SKHistogramNode<f64> = SKHistogramNode::leaf(7, 0.5);
    assert!(leaf.is_leaf());
    assert_eq!(leaf.sample_count(), 7);
    assert!((leaf.value() - 0.5).abs() < 1e-12);
}
