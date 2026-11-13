//! Walking one tree of an `ai.onnx.ml` ensemble.

use std::collections::BTreeMap;

use serde_onnx::ml::TreeNodes;

/// The distinct tree identifiers, in ascending order.
pub(crate) fn distinct_trees(nodes: &TreeNodes) -> Vec<i64> {
    let mut trees: Vec<i64> = nodes.treeids.clone();
    trees.sort_unstable();
    trees.dedup();
    trees
}

/// Walk one tree for one row and return the leaf node id.
pub(crate) fn walk(nodes: &TreeNodes, tree: i64, features: &[f64]) -> i64 {
    let index: BTreeMap<(i64, i64), usize> = nodes
        .nodeids
        .iter()
        .enumerate()
        .map(|(position, node)| ((nodes.treeids[position], *node), position))
        .collect();
    let mut current = nodes
        .nodeids
        .iter()
        .zip(nodes.treeids.iter())
        .filter(|(_, owner)| **owner == tree)
        .map(|(node, _)| *node)
        .min()
        .unwrap_or(0);
    loop {
        let Some(position) = index.get(&(tree, current)).copied() else {
            return current;
        };
        let mode = nodes.modes[position].as_str();
        if mode == "LEAF" {
            return current;
        }
        let feature = features
            .get(nodes.featureids[position] as usize)
            .copied()
            .unwrap_or(0.0);
        let threshold = f64::from(
            nodes
                .values
                .as_ref()
                .map_or(0.0, |values| values.get(position).copied().unwrap_or(0.0)),
        );
        let goes_true = if feature.is_nan() {
            nodes
                .missing_value_tracks_true
                .as_ref()
                .and_then(|flags| flags.get(position))
                .is_some_and(|flag| *flag != 0)
        } else {
            compare(mode, feature, threshold)
        };
        current = if goes_true {
            nodes.truenodeids[position]
        } else {
            nodes.falsenodeids[position]
        };
    }
}

fn compare(mode: &str, feature: f64, threshold: f64) -> bool {
    match mode {
        "BRANCH_LEQ" => feature <= threshold,
        "BRANCH_LT" => feature < threshold,
        "BRANCH_GTE" => feature >= threshold,
        "BRANCH_GT" => feature > threshold,
        "BRANCH_EQ" => feature == threshold,
        "BRANCH_NEQ" => feature != threshold,
        _ => false,
    }
}
