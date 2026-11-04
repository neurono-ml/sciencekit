//! Target canonicalization and per-head binary targets for the SGD classifier
//! (spec `stochastic-gradient-classifier`).

use sciencekit_common::sk_canonicalize_labels;
use sciencekit_common::{SKError, SKFloat, SKLabelTable, SKTargetView};

use crate::logistic::{sk_canonicalize_against, sk_canonicalize_targets};
use crate::sgd::SKSGDClassifierLoss;
use crate::sgd::SKSGDLoss;

/// Canonicalize integer and nominal targets through [`SKLabelTable`].
pub(crate) fn sk_canonicalize_classifier_targets<F: SKFloat>(
    targets: SKTargetView<'_, F>,
) -> Result<(Vec<usize>, SKLabelTable), SKError> {
    sk_canonicalize_targets(targets)
}

/// Canonicalize targets against the fitted class table, rejecting a new label.
pub(crate) fn sk_canonicalize_against_classes<F: SKFloat>(
    classes: &SKLabelTable,
    targets: SKTargetView<'_, F>,
) -> Result<Vec<usize>, SKError> {
    sk_canonicalize_against(classes, targets)
}

/// Reject a target set with fewer than two distinct classes.
pub(crate) fn sk_validate_number_of_classes(classes: &SKLabelTable) -> Result<(), SKError> {
    if classes.number_of_classes() < 2 {
        return Err(SKError::InvalidHyperparameter {
            name: "targets",
            reason: "a classifier requires at least two distinct classes".to_string(),
        });
    }
    Ok(())
}

/// The number of one-vs-rest heads: one for a binary problem, one per class
/// otherwise (spec decision 10).
pub(crate) fn sk_number_of_heads(number_of_classes: usize) -> usize {
    if number_of_classes == 2 {
        1
    } else {
        number_of_classes
    }
}

/// The canonical class each head scores positively.
///
/// A binary head scores the second class; a one-vs-rest head scores its own
/// class.
pub(crate) fn sk_positive_classes(number_of_classes: usize) -> Vec<usize> {
    if number_of_classes == 2 {
        vec![1]
    } else {
        (0..number_of_classes).collect()
    }
}

/// The `±1` targets of one head.
pub(crate) fn sk_head_targets<F: SKFloat>(
    class_indices: &[usize],
    positive_class: usize,
) -> Vec<F> {
    class_indices
        .iter()
        .map(|&index| {
            if index == positive_class {
                F::one()
            } else {
                -F::one()
            }
        })
        .collect()
}

/// The mean absolute slope at a zero predictor for one head's targets.
pub(crate) fn sk_initial_slope_scale<F: SKFloat>(loss: SKSGDClassifierLoss, targets: &[F]) -> F {
    if targets.is_empty() {
        return F::one();
    }
    let total = targets.iter().fold(F::zero(), |accumulator, &target| {
        accumulator + loss.slope(F::zero(), target).abs()
    });
    total / F::from(targets.len()).expect("the target count fits in the scalar")
}

/// A float-encoded class label rounded to its integer representation.
pub(crate) fn sk_integer_label<F: SKFloat>(value: F) -> i64 {
    value.round().to_i64().unwrap_or(0)
}

/// Build the label table of a stream whose labels are integer-coded floats.
pub(crate) fn sk_label_table_from_integers(labels: &[i64]) -> SKLabelTable {
    let owned: Vec<String> = labels.iter().map(i64::to_string).collect();
    let borrowed: Vec<&str> = owned.iter().map(String::as_str).collect();
    sk_canonicalize_labels(&borrowed).1
}
