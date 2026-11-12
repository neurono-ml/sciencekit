//! Shared fit helpers for the histogram boosters: importance aggregation and
//! early-stopping row splits.

use sciencekit_common::SKFloat;

use crate::boosting::early_stopping::{
    SKEarlyStoppingConfig, sk_split_classification_validation_rows,
    sk_split_regression_validation_rows,
};
use crate::histogram_boosting::growth::SKHistogramTree;

/// Split rows into a training/validation pair for early stopping, falling back
/// to fitting on every row when a held-out split is impossible.
pub(super) fn sk_validation_rows<F: SKFloat>(
    row_count: usize,
    class_indices: Option<&[usize]>,
    number_of_classes: usize,
    early_stopping: Option<SKEarlyStoppingConfig<F>>,
) -> (Vec<usize>, Option<Vec<usize>>) {
    let Some(stopping) = early_stopping else {
        return ((0..row_count).collect(), None);
    };
    let split = match class_indices {
        Some(class_indices) => sk_split_classification_validation_rows(
            class_indices,
            number_of_classes,
            stopping.validation_fraction,
            stopping.random_seed,
        ),
        None => sk_split_regression_validation_rows(
            row_count,
            stopping.validation_fraction,
            stopping.random_seed,
        ),
    };
    match split {
        Ok((training_rows, validation_rows)) => (training_rows, Some(validation_rows)),
        Err(_) => ((0..row_count).collect(), None),
    }
}

/// Aggregate per-stage tree importances weighted by each tree's root sample
/// count, normalizing to sum to one when a split occurred.
pub(super) fn sk_aggregate_histogram_importances<F: SKFloat>(
    stage_importances: &[Vec<F>],
    trees: &[&SKHistogramTree<F>],
    feature_count: usize,
) -> Vec<F> {
    let mut totals = vec![F::zero(); feature_count];
    stage_importances
        .iter()
        .zip(trees.iter())
        .for_each(|(importances, tree)| {
            let weight = F::from(tree.nodes()[0].sample_count()).expect("count fits");
            totals
                .iter_mut()
                .zip(importances.iter())
                .for_each(|(accumulator, &importance)| {
                    *accumulator = *accumulator + weight * importance;
                });
        });
    let normalizer = totals.iter().fold(F::zero(), |sum, &value| sum + value);
    if normalizer > F::zero() {
        totals
            .iter_mut()
            .for_each(|value| *value = *value / normalizer);
    }
    totals
}
