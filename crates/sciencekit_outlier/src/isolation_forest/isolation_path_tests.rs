//! Companion tests for `c(n)` path length, leaf-size adjustment and expected-depth scoring.

use sciencekit_tree::{SKTree, SKTreeNode};

use super::score::{
    sk_anomaly_score_from_expected_depth, sk_average_path_length, sk_expected_path_length,
    sk_tree_path_length,
};

#[test]
fn isolation_path_average_length_matches_hand_computed_harmonic_values() {
    assert_eq!(sk_average_path_length(0), 0.0);
    assert_eq!(sk_average_path_length(1), 0.0);
    assert!((sk_average_path_length(2) - 1.0).abs() < 1e-15);
    assert!((sk_average_path_length(3) - (5.0 / 3.0)).abs() < 1e-15);
    assert!((sk_average_path_length(4) - (13.0 / 6.0)).abs() < 1e-15);
    assert!((sk_average_path_length(5) - (77.0 / 30.0)).abs() < 1e-15);

    // Continuity at the exact-to-asymptotic boundary (m = 2048 vs m = 2049).
    let at_2049 = sk_average_path_length(2049);
    let at_2050 = sk_average_path_length(2050);
    assert!(at_2050 > at_2049);
    let exact_h_2049: f64 = (1..=2049).map(|term| 1.0 / term as f64).sum();
    let exact_c_2050 = 2.0 * exact_h_2049 - 2.0 * 2049.0 / 2050.0;
    assert!((at_2050 - exact_c_2050).abs() < 1e-14);
}

#[test]
fn isolation_path_tree_and_forest_reduction_match_hand_built_trees() {
    // Tree 1: root splits feature 0 at 5.0; left leaf has 4 unsplit samples, right leaf has 1 isolated sample.
    let tree_one = SKTree::new(
        vec![
            SKTreeNode::split(0, 5.0_f64, 1, 2, 5, 0.0),
            SKTreeNode::leaf(4, 0.0, Vec::new()),
            SKTreeNode::leaf(1, 0.0, Vec::new()),
        ],
        1,
    );
    // Tree 2: root splits feature 0 at 2.0; left leaf has 2 samples, right splits at 8.0 into 2 samples and 1 sample.
    let tree_two = SKTree::new(
        vec![
            SKTreeNode::split(0, 2.0_f64, 1, 2, 5, 0.0),
            SKTreeNode::leaf(2, 0.0, Vec::new()),
            SKTreeNode::split(0, 8.0_f64, 3, 4, 3, 0.0),
            SKTreeNode::leaf(2, 0.0, Vec::new()),
            SKTreeNode::leaf(1, 0.0, Vec::new()),
        ],
        1,
    );

    // Outlier x = 10.0:
    // - Tree 1: depth 1 + c(1) = 1.0
    // - Tree 2: depth 2 + c(1) = 2.0
    // Expected depth = 1.5
    let outlier_one = sk_tree_path_length(&tree_one, |_| 10.0);
    let outlier_two = sk_tree_path_length(&tree_two, |_| 10.0);
    assert!((outlier_one - 1.0).abs() < 1e-15);
    assert!((outlier_two - 2.0).abs() < 1e-15);

    let trees = [tree_one, tree_two];
    let outlier_expected = sk_expected_path_length(&trees, |_| 10.0);
    assert!((outlier_expected - 1.5).abs() < 1e-15);

    // Inlier x = 1.0:
    // - Tree 1: depth 1 + c(4) = 1 + 13/6 = 19/6
    // - Tree 2: depth 1 + c(2) = 1 + 1 = 2
    // Expected depth = (19/6 + 2) / 2 = 31/12
    let inlier_expected = sk_expected_path_length(&trees, |_| 1.0);
    assert!((inlier_expected - (31.0 / 12.0)).abs() < 1e-15);

    let outlier_score = sk_anomaly_score_from_expected_depth(outlier_expected, 5);
    let inlier_score = sk_anomaly_score_from_expected_depth(inlier_expected, 5);
    let expected_outlier_score = 2.0_f64.powf(-1.5 / (77.0 / 30.0));
    let expected_inlier_score = 2.0_f64.powf(-(31.0 / 12.0) / (77.0 / 30.0));

    assert!((outlier_score - expected_outlier_score).abs() < 1e-15);
    assert!((inlier_score - expected_inlier_score).abs() < 1e-15);
    assert!(outlier_score > inlier_score);
    assert!((0.0..=1.0).contains(&outlier_score));
    assert!((0.0..=1.0).contains(&inlier_score));
    assert!((sk_anomaly_score_from_expected_depth(0.0, 1) - 0.5).abs() < 1e-15);
}
