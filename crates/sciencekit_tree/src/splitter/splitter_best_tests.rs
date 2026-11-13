//! `SKSplitter::Best` tests: parallel evaluation and deterministic ties.

use ndarray::{Array2, array};

use super::{SKScanContext, SKSplitter, sk_best_split};
use crate::criterion::SKTreeCriterion;
use crate::design::SKDesign;

fn fixture() -> (Array2<f64>, Vec<usize>) {
    let features = array![
        [1.0, 10.0, 5.0],
        [2.0, 10.0, 5.0],
        [3.0, 20.0, 5.0],
        [4.0, 20.0, 5.0],
    ];
    (features, vec![0, 0, 1, 1])
}

fn context<'a>(classes: &'a [usize], weights: &'a [f64]) -> SKScanContext<'a, f64> {
    SKScanContext::Classification {
        class_indices: classes,
        sample_weights: weights,
        number_of_classes: 2,
        criterion: SKTreeCriterion::Gini,
    }
}

/// Equally good splits on features 0 and 1 resolve to the smaller feature
/// index, even though both are evaluated in parallel.
#[test]
fn best_split_breaks_ties_by_smaller_feature() {
    let (features, classes) = fixture();
    let design = SKDesign::Dense(features.view());
    let weights = [1.0; 4];
    let candidate = sk_best_split(
        &design,
        &[0, 1, 2, 3],
        &[0, 1, 2],
        1,
        context(&classes, &weights),
        SKSplitter::Best,
        7,
    )
    .expect("a split exists");
    assert_eq!(candidate.feature(), 0);
    assert_eq!(candidate.threshold(), 2.5);
}

/// Supplying the features in a different order yields the same candidate.
#[test]
fn best_split_is_independent_of_feature_order() {
    let (features, classes) = fixture();
    let design = SKDesign::Dense(features.view());
    let weights = [1.0; 4];
    let ordered = sk_best_split(
        &design,
        &[0, 1, 2, 3],
        &[0, 1, 2],
        1,
        context(&classes, &weights),
        SKSplitter::Best,
        7,
    );
    let reversed = sk_best_split(
        &design,
        &[0, 1, 2, 3],
        &[2, 1, 0],
        1,
        context(&classes, &weights),
        SKSplitter::Best,
        7,
    );
    assert_eq!(ordered, reversed);
}

/// Restricting the considered features restricts the chosen split.
#[test]
fn best_split_respects_the_considered_feature_set() {
    let (features, classes) = fixture();
    let design = SKDesign::Dense(features.view());
    let weights = [1.0; 4];
    let candidate = sk_best_split(
        &design,
        &[0, 1, 2, 3],
        &[1],
        1,
        context(&classes, &weights),
        SKSplitter::Best,
        7,
    )
    .expect("feature 1 separates the classes");
    assert_eq!(candidate.feature(), 1);
    assert_eq!(candidate.threshold(), 15.0);
}
