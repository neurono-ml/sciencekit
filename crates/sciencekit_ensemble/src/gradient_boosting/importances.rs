//! Sample-weighted aggregation and normalization of stage-tree feature
//! importances.

use sciencekit_common::SKFloat;

use crate::boosting::stage::SKFittedStageTree;

/// Aggregate feature importances across `stage_trees` weighted by each stage
/// tree's training `sample_count`, normalizing the vector to sum to `1` when at
/// least one split occurred.
pub(crate) fn sk_aggregate_stage_importances<'a, F, I>(
    stage_trees: I,
    feature_count: usize,
) -> Vec<F>
where
    F: SKFloat + 'a,
    I: IntoIterator<Item = &'a SKFittedStageTree<F>>,
{
    let mut totals = vec![F::zero(); feature_count];
    if feature_count == 0 {
        return totals;
    }
    stage_trees.into_iter().for_each(|stage_tree| {
        let weight = F::from(stage_tree.sample_count).expect("sample count fits in scalar");
        totals
            .iter_mut()
            .zip(stage_tree.feature_importances.iter())
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
