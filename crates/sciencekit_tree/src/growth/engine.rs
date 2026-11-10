//! Recursive partitioning with a flat node array.

use sciencekit_common::{SK_RANDOM_DOMAIN_TREE, SKFloat, sk_derive_seed};

use super::{SKGrowthConfig, SKTreeCriterion, feature_selection::sk_select_features};
use crate::design::SKDesign;
use crate::node::{SKTree, SKTreeNode};
use crate::splitter::sk_best_split;

/// Grow a tree and its normalized impurity-based feature importances.
pub(crate) fn sk_grow_tree<F: SKFloat, C: SKTreeCriterion<F>>(
    criterion: &C,
    design: &SKDesign<'_, F>,
    config: &SKGrowthConfig<F>,
) -> (SKTree<F>, Vec<F>) {
    let feature_count = design.columns();
    let rows: Vec<usize> = (0..design.rows()).collect();
    let mut nodes = vec![SKTreeNode::leaf(rows.len(), F::zero(), Vec::new())];
    let mut importances = vec![F::zero(); feature_count];
    let mut tasks = vec![(0usize, rows, 0usize)];

    while let Some((node_index, sample_indices, depth)) = tasks.pop() {
        let impurity = criterion.impurity(&sample_indices);
        let can_descend = config
            .maximum_depth
            .is_none_or(|maximum_depth| depth < maximum_depth);
        let large_enough = sample_indices.len() >= config.minimum_samples_split
            && sample_indices.len() >= config.minimum_samples_leaf.saturating_mul(2);
        if can_descend && large_enough {
            let features = sk_select_features(config, feature_count, node_index);
            let node_seed =
                sk_derive_seed(config.random_seed, SK_RANDOM_DOMAIN_TREE, node_index as u64);
            let candidate = sk_best_split(
                design,
                &sample_indices,
                &features,
                config.minimum_samples_leaf,
                criterion.scan_context(),
                config.splitter,
                node_seed,
            );
            if let Some(candidate) = candidate {
                let (left, right) =
                    design.partition(&sample_indices, candidate.feature(), candidate.threshold());
                if left.len() >= config.minimum_samples_leaf
                    && right.len() >= config.minimum_samples_leaf
                {
                    let left_index = nodes.len();
                    let right_index = left_index + 1;
                    nodes.push(SKTreeNode::leaf(left.len(), F::zero(), Vec::new()));
                    nodes.push(SKTreeNode::leaf(right.len(), F::zero(), Vec::new()));
                    nodes[node_index] = SKTreeNode::split(
                        candidate.feature(),
                        candidate.threshold(),
                        left_index,
                        right_index,
                        sample_indices.len(),
                        impurity,
                    );
                    importances[candidate.feature()] = importances[candidate.feature()]
                        + F::from(sample_indices.len()).expect("count fits")
                            * candidate.impurity_decrease();
                    tasks.push((right_index, right, depth + 1));
                    tasks.push((left_index, left, depth + 1));
                    continue;
                }
            }
        }
        nodes[node_index] = SKTreeNode::leaf(
            sample_indices.len(),
            impurity,
            criterion.leaf_output(&sample_indices),
        );
    }

    let total: F = importances
        .iter()
        .fold(F::zero(), |sum, importance| sum + *importance);
    if total > F::zero() {
        for importance in &mut importances {
            *importance = *importance / total;
        }
    }
    (SKTree::new(nodes, feature_count), importances)
}
