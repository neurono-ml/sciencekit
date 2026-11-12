//! Feature-subsampling tests (spec `decision-trees`).

use super::{SKGrowthConfig, sk_select_features};
use crate::splitter::{SKMaximumFeatures, SKSplitter};

fn config(maximum_features: SKMaximumFeatures<f64>, seed: u64) -> SKGrowthConfig<f64> {
    SKGrowthConfig {
        maximum_depth: None,
        minimum_samples_split: 2,
        minimum_samples_leaf: 1,
        maximum_features,
        splitter: SKSplitter::Best,
        random_seed: seed,
    }
}

/// Every policy resolves against a nine-feature problem as documented.
#[test]
fn policies_resolve_the_documented_feature_count() {
    assert_eq!(SKMaximumFeatures::<f64>::All.resolved_count(9), 9);
    assert_eq!(SKMaximumFeatures::<f64>::Count(4).resolved_count(9), 4);
    assert_eq!(SKMaximumFeatures::<f64>::Fraction(0.5).resolved_count(9), 4);
    assert_eq!(SKMaximumFeatures::<f64>::Sqrt.resolved_count(9), 3);
    assert_eq!(SKMaximumFeatures::<f64>::Log2.resolved_count(9), 3);
}

/// A count larger than the design is clamped to the available features, and a
/// zero design resolves to zero.
#[test]
fn counts_are_clamped_to_the_available_features() {
    assert_eq!(SKMaximumFeatures::<f64>::Count(20).resolved_count(5), 5);
    assert_eq!(SKMaximumFeatures::<f64>::Fraction(2.0).resolved_count(5), 5);
    assert_eq!(SKMaximumFeatures::<f64>::All.resolved_count(0), 0);
}

/// The same seed and node select the same features; the selection is sorted
/// for a deterministic scan order.
#[test]
fn seeded_selection_is_repeatable_and_sorted() {
    let constraints = config(SKMaximumFeatures::Sqrt, 17);
    let first = sk_select_features(&constraints, 9, 3);
    let second = sk_select_features(&constraints, 9, 3);
    assert_eq!(first, second);
    assert_eq!(first.len(), 3);
    assert!(first.windows(2).all(|pair| pair[0] < pair[1]));
}

/// Different nodes resolve different feature subsets from the same seed.
#[test]
fn different_nodes_select_different_subsets() {
    let constraints = config(SKMaximumFeatures::Count(2), 17);
    let selections: std::collections::BTreeSet<Vec<usize>> = (0..16)
        .map(|node| sk_select_features(&constraints, 8, node))
        .collect();
    assert!(selections.len() > 1);
}

/// `All` returns every feature in order without drawing from the stream.
#[test]
fn all_features_are_considered_in_order() {
    let constraints = config(SKMaximumFeatures::All, 0);
    assert_eq!(sk_select_features(&constraints, 4, 0), vec![0, 1, 2, 3]);
}
