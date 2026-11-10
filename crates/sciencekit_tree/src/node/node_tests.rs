//! Companion tests for the flat tree structure (spec `decision-trees`).

use sciencekit_common::SKFloat;

use super::{SKTree, SKTreeNode};

/// Build the hand-checkable three-node tree used by the accessor and traversal
/// tests: a root splitting feature 0 at 2.5, with two pure leaves.
fn three_node_tree<F: SKFloat>() -> SKTree<F> {
    let zero = F::zero();
    let left = SKTreeNode::leaf(2, zero, vec![F::one(), zero]);
    let right = SKTreeNode::leaf(2, zero, vec![zero, F::one()]);
    let root = SKTreeNode::split(
        0,
        F::from(2.5).expect("2.5 fits"),
        1,
        2,
        4,
        F::from(0.5).expect("0.5 fits"),
    );
    let tree = SKTree::new(vec![root, left, right], 1);
    assert_eq!(tree.node_count(), 3);
    assert_eq!(tree.depth(), 1);
    tree
}

/// An internal node exposes its feature, threshold, children, count and
/// impurity, and reports itself as non-leaf.
#[test]
fn split_node_exposes_its_split_contract() {
    let tree = three_node_tree::<f64>();
    let root = tree.node(tree.root_index()).expect("root exists");
    assert!(!root.is_leaf());
    assert_eq!(root.split_feature(), Some(0));
    assert_eq!(root.threshold(), Some(2.5));
    assert_eq!(root.left_child(), Some(1));
    assert_eq!(root.right_child(), Some(2));
    assert_eq!(root.sample_count(), 4);
    assert_eq!(root.impurity(), 0.5);
    assert!(root.value().is_empty());
}

/// A leaf exposes its learned distribution and carries no split fields.
#[test]
fn leaf_node_exposes_its_value() {
    let tree = three_node_tree::<f64>();
    let left = tree.node(1).expect("left leaf exists");
    assert!(left.is_leaf());
    assert_eq!(left.split_feature(), None);
    assert_eq!(left.threshold(), None);
    assert_eq!(left.left_child(), None);
    assert_eq!(left.right_child(), None);
    assert_eq!(left.sample_count(), 2);
    assert_eq!(left.value(), &[1.0, 0.0]);
}

/// Traversal follows `value <= threshold` to the left and returns leaf indices.
#[test]
fn traversal_reaches_the_expected_leaf() {
    let tree = three_node_tree::<f64>();
    assert_eq!(tree.leaf_index_for(|_| 1.0), 1);
    assert_eq!(tree.leaf_index_for(|_| 2.5), 1);
    assert_eq!(tree.leaf_index_for(|_| 2.500_001), 2);
}

/// A single-node tree is a root leaf at depth zero.
#[test]
fn single_node_tree_is_a_root_leaf() {
    let tree: SKTree<f32> = SKTree::new(vec![SKTreeNode::leaf(7, 0.0, vec![3.0])], 2);
    assert_eq!(tree.depth(), 0);
    assert_eq!(tree.leaf_index_for(|_| unreachable!()), 0);
    assert_eq!(tree.feature_count(), 2);
    assert_eq!(tree.node(9), None);
}
