//! Growth-constraint tests (spec `decision-trees`).

use super::{SKGrowthConfig, SKTreeCriterion, sk_grow_tree};
use crate::criterion::SKTreeCriterion as SKConfiguredCriterion;
use crate::design::SKDesign;
use crate::impurity::sk_gini_impurity;
use crate::splitter::{SKMaximumFeatures, SKScanContext, SKSplitter};

/// A minimal classification criterion used to exercise the engine before the
/// estimator modules exist; it mirrors the classifier's contract exactly.
struct SKTestCriterion<'a> {
    classes: &'a [usize],
    weights: &'a [f64],
    number_of_classes: usize,
}

impl SKTreeCriterion<f64> for SKTestCriterion<'_> {
    fn impurity(&self, sample_indices: &[usize]) -> f64 {
        let mut counts = vec![0.0; self.number_of_classes];
        for &row in sample_indices {
            counts[self.classes[row]] += self.weights[row];
        }
        sk_gini_impurity(&counts)
    }

    fn leaf_output(&self, sample_indices: &[usize]) -> Vec<f64> {
        let mut counts = vec![0.0; self.number_of_classes];
        for &row in sample_indices {
            counts[self.classes[row]] += self.weights[row];
        }
        let total: f64 = counts.iter().sum();
        counts.iter().map(|count| count / total).collect()
    }

    fn scan_context(&self) -> SKScanContext<'_, f64> {
        SKScanContext::Classification {
            class_indices: self.classes,
            sample_weights: self.weights,
            number_of_classes: self.number_of_classes,
            criterion: SKConfiguredCriterion::Gini,
        }
    }
}

fn config(maximum_depth: Option<usize>) -> SKGrowthConfig<f64> {
    SKGrowthConfig {
        maximum_depth,
        minimum_samples_split: 2,
        minimum_samples_leaf: 1,
        maximum_features: SKMaximumFeatures::All,
        splitter: SKSplitter::Best,
        random_seed: 0,
    }
}

fn grow(
    features: ndarray::ArrayView2<'_, f64>,
    classes: &[usize],
    weights: &[f64],
    config: &SKGrowthConfig<f64>,
) -> crate::node::SKTree<f64> {
    let criterion = SKTestCriterion {
        classes,
        weights,
        number_of_classes: 2,
    };
    let design = SKDesign::Dense(features);
    sk_grow_tree(&criterion, &design, config).0
}

fn separable() -> ndarray::Array2<f64> {
    ndarray::array![[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]]
}

/// A depth limit of zero leaves the root as a leaf even though a split exists.
#[test]
fn maximum_depth_zero_stops_at_the_root() {
    let features = separable();
    let tree = grow(
        features.view(),
        &[0, 0, 0, 1, 1, 1],
        &[1.0; 6],
        &config(Some(0)),
    );
    assert_eq!(tree.node_count(), 1);
    assert!(tree.nodes()[0].is_leaf());
}

/// A depth limit of one yields two leaves and no grandchildren.
#[test]
fn maximum_depth_one_stops_after_the_root_split() {
    let features = separable();
    let tree = grow(
        features.view(),
        &[0, 0, 0, 1, 1, 1],
        &[1.0; 6],
        &config(Some(1)),
    );
    assert_eq!(tree.node_count(), 3);
    assert_eq!(tree.depth(), 1);
}

/// A minimum split size above the node's sample count prevents the split.
#[test]
fn minimum_samples_split_stops_growth() {
    let features = separable();
    let mut constraints = config(None);
    constraints.minimum_samples_split = 7;
    let tree = grow(
        features.view(),
        &[0, 0, 0, 1, 1, 1],
        &[1.0; 6],
        &constraints,
    );
    assert_eq!(tree.node_count(), 1);
}

/// A minimum leaf size larger than a side of every candidate prevents the
/// split.
#[test]
fn minimum_samples_leaf_stops_growth() {
    let features = separable();
    let mut constraints = config(None);
    constraints.minimum_samples_leaf = 4;
    let tree = grow(
        features.view(),
        &[0, 0, 0, 1, 1, 1],
        &[1.0; 6],
        &constraints,
    );
    assert_eq!(tree.node_count(), 1);
}
