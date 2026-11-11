//! Path-length correction `c(n)`, expected depth reduction and anomaly score.

#![cfg_attr(not(test), allow(dead_code))]

use sciencekit_common::SKFloat;
use sciencekit_tree::SKTree;

/// Euler-Mascheroni constant used in the large-`n` harmonic expansion.
const EULER_MASCHERONI: f64 = 0.577_215_664_901_532_9;

/// Exact harmonic-sum cutoff; above this, Euler-Maclaurin agrees to all 53 `f64` bits.
const EXACT_HARMONIC_CUTOFF: usize = 2048;

/// Harmonic number `H(order) = sum_{k=1..=order} 1 / k`.
fn harmonic_number(order: usize) -> f64 {
    if order == 0 {
        return 0.0;
    }
    if order <= EXACT_HARMONIC_CUTOFF {
        return (1..=order).map(|term| 1.0 / term as f64).sum();
    }
    let real_order = order as f64;
    let inverse = 1.0 / real_order;
    let inverse_squared = inverse * inverse;
    let inverse_fourth = inverse_squared * inverse_squared;
    real_order.ln() + EULER_MASCHERONI + 0.5 * inverse - inverse_squared / 12.0
        + inverse_fourth / 120.0
}

/// Average path length `c(n) = 2 H(n - 1) - 2 (n - 1) / n` of an unsuccessful BST search.
pub(crate) fn sk_average_path_length(sample_count: usize) -> f64 {
    if sample_count <= 1 {
        return 0.0;
    }
    let predecessor = sample_count - 1;
    2.0 * harmonic_number(predecessor) - (2.0 * predecessor as f64) / (sample_count as f64)
}

/// Path length `h(x) = edge_depth + c(leaf_size)` for one sample in one isolation tree.
pub(crate) fn sk_tree_path_length<F: SKFloat>(
    tree: &SKTree<F>,
    feature_value: impl Fn(usize) -> F,
) -> f64 {
    let (leaf_index, edge_depth) = tree.leaf_and_depth_for(feature_value);
    let leaf_sample_count = tree
        .node(leaf_index)
        .map_or(1, |leaf_node| leaf_node.sample_count());
    edge_depth as f64 + sk_average_path_length(leaf_sample_count)
}

/// Expected path length `E[h(x)]` across all isolation trees in a forest.
pub(crate) fn sk_expected_path_length<F: SKFloat>(
    trees: &[SKTree<F>],
    feature_value: impl Fn(usize) -> F + Copy,
) -> f64 {
    if trees.is_empty() {
        return 0.0;
    }
    let total_depth: f64 = trees
        .iter()
        .map(|tree| sk_tree_path_length(tree, feature_value))
        .sum();
    total_depth / (trees.len() as f64)
}

/// Liu-Ting-Zhou anomaly score `s(x, n) = 2^(-E[h(x)] / c(n))` in `[0, 1]`.
pub(crate) fn sk_anomaly_score_from_expected_depth(
    expected_depth: f64,
    subsample_size: usize,
) -> f64 {
    let normalizer = sk_average_path_length(subsample_size);
    if normalizer <= 0.0 {
        return 0.5;
    }
    2.0_f64.powf(-expected_depth / normalizer).clamp(0.0, 1.0)
}
