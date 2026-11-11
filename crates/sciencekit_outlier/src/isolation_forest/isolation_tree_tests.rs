//! Companion tests for random isolation tree splits and growth limits.

use ndarray::{Array2, array};
use sciencekit_common::SKRandomGenerator;

use super::tree::{sk_build_dense_isolation_tree, sk_isolation_depth_limit};

#[test]
fn isolation_tree_depth_limit_follows_ceil_log2() {
    assert_eq!(sk_isolation_depth_limit(0), 0);
    assert_eq!(sk_isolation_depth_limit(1), 0);
    assert_eq!(sk_isolation_depth_limit(2), 1);
    assert_eq!(sk_isolation_depth_limit(3), 2);
    assert_eq!(sk_isolation_depth_limit(4), 2);
    assert_eq!(sk_isolation_depth_limit(5), 3);
    assert_eq!(sk_isolation_depth_limit(256), 8);
    assert_eq!(sk_isolation_depth_limit(257), 9);
}

#[test]
fn isolation_tree_single_sample_becomes_immediate_leaf() {
    let single = array![[3.5_f64, -1.2]];
    let mut generator = SKRandomGenerator::new(7);
    let tree = sk_build_dense_isolation_tree(single.view(), &[0, 1], 8, &mut generator);

    assert_eq!(tree.node_count(), 1);
    assert_eq!(tree.depth(), 0);
    let root = tree.node(tree.root_index()).unwrap();
    assert!(root.is_leaf());
    assert_eq!(root.sample_count(), 1);
    let (leaf_index, edge_depth) = tree.leaf_and_depth_for(|feature| single[[0, feature]]);
    assert_eq!(leaf_index, 0);
    assert_eq!(edge_depth, 0);
}

#[test]
fn isolation_tree_splits_stay_within_observed_feature_bounds_and_cap_depth() {
    let data = array![
        [0.0_f64, 10.0],
        [1.0, 20.0],
        [2.0, 30.0],
        [3.0, 40.0],
        [4.0, 50.0],
        [5.0, 60.0],
        [6.0, 70.0],
        [100.0, 800.0]
    ];
    let mut generator = SKRandomGenerator::new(42);
    let depth_limit = sk_isolation_depth_limit(data.nrows());
    assert_eq!(depth_limit, 3);
    let tree = sk_build_dense_isolation_tree(data.view(), &[0, 1], depth_limit, &mut generator);

    assert!(tree.depth() <= depth_limit);
    assert!(tree.node_count() >= 3);

    // Verify every internal node's threshold lies within the observed range of the
    // rows reaching that node, and every child strictly partitions those rows.
    let mut stack: Vec<(usize, Vec<usize>)> =
        vec![(tree.root_index(), (0..data.nrows()).collect())];
    while let Some((node_index, node_rows)) = stack.pop() {
        let node = tree.node(node_index).unwrap();
        assert_eq!(node.sample_count(), node_rows.len());
        if !node.is_leaf() {
            let feature = node.split_feature().unwrap();
            let threshold = node.threshold().unwrap();
            let observed_minimum = node_rows
                .iter()
                .map(|&row| data[[row, feature]])
                .fold(f64::INFINITY, f64::min);
            let observed_maximum = node_rows
                .iter()
                .map(|&row| data[[row, feature]])
                .fold(f64::NEG_INFINITY, f64::max);
            assert!(observed_minimum < observed_maximum);
            assert!(threshold >= observed_minimum && threshold < observed_maximum);

            let (left_rows, right_rows): (Vec<usize>, Vec<usize>) = node_rows
                .into_iter()
                .partition(|&row| data[[row, feature]] <= threshold);
            assert!(!left_rows.is_empty());
            assert!(!right_rows.is_empty());
            stack.push((node.left_child().unwrap(), left_rows));
            stack.push((node.right_child().unwrap(), right_rows));
        }
    }
}

#[test]
fn isolation_tree_skips_constant_features_and_stops_on_identical_rows() {
    let identical = Array2::from_elem((5, 3), 2.5_f64);
    let mut generator = SKRandomGenerator::new(9);
    let tree_identical =
        sk_build_dense_isolation_tree(identical.view(), &[0, 1, 2], 4, &mut generator);
    assert_eq!(tree_identical.node_count(), 1);
    assert_eq!(tree_identical.node(0).unwrap().sample_count(), 5);

    // Feature 0 and 1 are constant; only feature 2 varies and must isolate both points.
    let one_varying = array![[0.0_f64, 1.0, -5.0], [0.0, 1.0, 5.0]];
    let mut second_generator = SKRandomGenerator::new(9);
    let tree_varying =
        sk_build_dense_isolation_tree(one_varying.view(), &[0, 1, 2], 4, &mut second_generator);
    assert_eq!(tree_varying.node_count(), 3);
    assert_eq!(tree_varying.node(0).unwrap().split_feature(), Some(2));
}
