//! Dense split-scan tests (spec `decision-trees`).

use ndarray::{Array2, array};

use super::{SKScanContext, SKThresholdChoice, sk_scan_feature};
use crate::criterion::SKTreeCriterion;
use crate::design::SKDesign;

/// The hand-checkable two-feature fixture: feature 0 and feature 1 each
/// separate the two classes, while feature 2 is constant.
fn classification_fixture() -> (Array2<f64>, Vec<usize>) {
    let features = array![
        [1.0, 10.0, 5.0],
        [2.0, 10.0, 5.0],
        [3.0, 20.0, 5.0],
        [4.0, 20.0, 5.0],
    ];
    (features, vec![0, 0, 1, 1])
}

fn classification_context<'a>(
    class_indices: &'a [usize],
    weights: &'a [f64],
) -> SKScanContext<'a, f64> {
    SKScanContext::Classification {
        class_indices,
        sample_weights: weights,
        number_of_classes: 2,
        criterion: SKTreeCriterion::Gini,
    }
}

/// The scan recovers the class-separating threshold and its impurity decrease.
#[test]
fn dense_scan_recovers_the_separating_split() {
    let (features, classes) = classification_fixture();
    let design = SKDesign::Dense(features.view());
    let weights = [1.0; 4];
    let candidate = sk_scan_feature(
        &design,
        &[0, 1, 2, 3],
        0,
        1,
        classification_context(&classes, &weights),
        SKThresholdChoice::Best,
    )
    .expect("a split exists");
    assert_eq!(candidate.feature(), 0);
    assert_eq!(candidate.threshold(), 2.5);
    assert!((candidate.impurity_decrease() - 0.5).abs() < 1e-12);
}

/// The second separating feature is also found with the same decrease.
#[test]
fn dense_scan_evaluates_every_feature_independently() {
    let (features, classes) = classification_fixture();
    let design = SKDesign::Dense(features.view());
    let weights = [1.0; 4];
    let candidate = sk_scan_feature(
        &design,
        &[0, 1, 2, 3],
        1,
        1,
        classification_context(&classes, &weights),
        SKThresholdChoice::Best,
    )
    .expect("a split exists");
    assert_eq!(candidate.threshold(), 15.0);
    assert!((candidate.impurity_decrease() - 0.5).abs() < 1e-12);
}

/// A constant feature offers no valid threshold.
#[test]
fn dense_scan_finds_no_split_on_a_constant_feature() {
    let (features, classes) = classification_fixture();
    let design = SKDesign::Dense(features.view());
    let weights = [1.0; 4];
    assert!(
        sk_scan_feature(
            &design,
            &[0, 1, 2, 3],
            2,
            1,
            classification_context(&classes, &weights),
            SKThresholdChoice::Best,
        )
        .is_none()
    );
}

/// A minimum leaf larger than a side of the only good split rejects it.
#[test]
fn dense_scan_honors_the_minimum_leaf() {
    let (features, classes) = classification_fixture();
    let design = SKDesign::Dense(features.view());
    let weights = [1.0; 4];
    assert!(
        sk_scan_feature(
            &design,
            &[0, 1, 2, 3],
            0,
            3,
            classification_context(&classes, &weights),
            SKThresholdChoice::Best,
        )
        .is_none()
    );
}
