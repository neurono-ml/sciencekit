//! Aggregation and normalization of per-tree impurity-based feature
//! importances.

use sciencekit_common::SKFloat;
use sciencekit_tree::SKDecisionTreeRegressorModel;

use super::classifier_model::SKClassifierTreeMember;

/// Aggregate per-tree classifier feature importances into a normalized vector.
pub(super) fn sk_classifier_importances<F: SKFloat>(
    members: &[SKClassifierTreeMember<F>],
    feature_count: usize,
) -> Vec<F> {
    let mut totals = vec![F::zero(); feature_count];
    members.iter().for_each(|member| {
        if let SKClassifierTreeMember::Fitted { model, .. } = member {
            totals
                .iter_mut()
                .zip(model.feature_importances().iter())
                .for_each(|(target, &value)| *target = *target + value);
        }
    });
    sk_normalize_mean(totals, members.len())
}

/// Aggregate per-tree regressor feature importances into a normalized vector.
pub(super) fn sk_regressor_importances<F: SKFloat>(
    trees: &[SKDecisionTreeRegressorModel<F>],
    feature_count: usize,
) -> Vec<F> {
    let mut totals = vec![F::zero(); feature_count];
    trees.iter().for_each(|tree| {
        totals
            .iter_mut()
            .zip(tree.feature_importances().iter())
            .for_each(|(target, &value)| *target = *target + value);
    });
    sk_normalize_mean(totals, trees.len())
}

fn sk_normalize_mean<F: SKFloat>(mut totals: Vec<F>, tree_count: usize) -> Vec<F> {
    if tree_count == 0 || totals.is_empty() {
        return totals;
    }
    let divisor = F::from(tree_count).expect("tree count fits in scalar");
    totals
        .iter_mut()
        .for_each(|value| *value = *value / divisor);
    let sum = totals.iter().fold(F::zero(), |acc, &value| acc + value);
    if sum > F::zero() {
        totals.iter_mut().for_each(|value| *value = *value / sum);
    }
    totals
}
