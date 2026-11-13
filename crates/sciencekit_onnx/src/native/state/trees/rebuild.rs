//! Rebuilding per-tree parallel arrays from the flat ONNX-ML representation.

use std::collections::BTreeMap;

use serde_onnx::ml::TreeNodes;

use crate::native::error::SKOnnxNativeError;

/// One fitted tree rebuilt from the parallel ONNX arrays.
pub(crate) struct Tree {
    pub(crate) split_feature: Vec<i64>,
    pub(crate) threshold: Vec<f64>,
    pub(crate) left: Vec<i64>,
    pub(crate) right: Vec<i64>,
    pub(crate) values: Vec<Vec<f64>>,
}

#[derive(Clone, Copy)]
struct RawNode {
    node_id: i64,
    flat: usize,
}

/// Reject a tree whose routing the native model cannot represent.
///
/// The native tree always sends `feature <= threshold` left and never routes a
/// missing value, so a graph using another branch mode or
/// `missing_value_tracks_true` would silently predict different values.
pub(crate) fn validate_semantics(
    nodes: &TreeNodes,
    operator: &str,
) -> Result<(), SKOnnxNativeError> {
    let missing = nodes.missing_value_tracks_true.as_deref().unwrap_or(&[]);
    for (position, mode) in nodes.modes.iter().enumerate() {
        if mode == "LEAF" {
            continue;
        }
        if mode != "BRANCH_LEQ" {
            return Err(SKOnnxNativeError::UnsupportedOperator {
                operator: operator.to_string(),
                reason: format!(
                    "the native tree routes with `feature <= threshold` only; branch mode `{mode}` is outside the subset"
                ),
            });
        }
        if missing
            .get(position)
            .is_some_and(|tracks_true| *tracks_true != 0)
        {
            return Err(SKOnnxNativeError::UnsupportedOperator {
                operator: operator.to_string(),
                reason: "the native tree rejects non-finite input, so `missing_value_tracks_true` is outside the subset"
                    .to_string(),
            });
        }
    }
    Ok(())
}

/// Group the flat arrays by tree, remap node ids to positions and read leaves.
pub(crate) fn rebuild<Value>(nodes: &TreeNodes, value_of: Value) -> Vec<Tree>
where
    Value: Fn(i64, i64) -> Vec<f64>,
{
    let mut grouped: BTreeMap<i64, Vec<RawNode>> = BTreeMap::new();
    for (flat, tree) in nodes.treeids.iter().enumerate() {
        grouped.entry(*tree).or_default().push(RawNode {
            node_id: nodes.nodeids[flat],
            flat,
        });
    }
    let mut trees = Vec::with_capacity(grouped.len());
    for (tree_id, mut members) in grouped {
        members.sort_unstable_by_key(|member| member.node_id);
        let positions: BTreeMap<i64, usize> = members
            .iter()
            .enumerate()
            .map(|(index, member)| (member.node_id, index))
            .collect();
        let mut tree = Tree {
            split_feature: Vec::with_capacity(members.len()),
            threshold: Vec::with_capacity(members.len()),
            left: Vec::with_capacity(members.len()),
            right: Vec::with_capacity(members.len()),
            values: Vec::with_capacity(members.len()),
        };
        for member in &members {
            let leaf = nodes
                .modes
                .get(member.flat)
                .is_some_and(|mode| mode == "LEAF");
            tree.split_feature.push(if leaf {
                -1
            } else {
                nodes.featureids[member.flat]
            });
            tree.threshold.push(if leaf {
                0.0
            } else {
                nodes
                    .values
                    .as_ref()
                    .map_or(0.0, |values| f64::from(values[member.flat]))
            });
            tree.left.push(if leaf {
                -1
            } else {
                *positions.get(&nodes.truenodeids[member.flat]).unwrap_or(&0) as i64
            });
            tree.right.push(if leaf {
                -1
            } else {
                *positions
                    .get(&nodes.falsenodeids[member.flat])
                    .unwrap_or(&0) as i64
            });
            tree.values.push(value_of(member.node_id, tree_id));
        }
        trees.push(tree);
    }
    trees
}

/// The distinct tree ids the ensemble declares, in ascending order.
pub(crate) fn distinct_tree_ids(nodes: &TreeNodes) -> Vec<i64> {
    let mut ids = nodes.treeids.clone();
    ids.sort_unstable();
    ids.dedup();
    ids
}

/// The target weight stored for one `(tree, node)` pair.
pub(crate) fn weight_at(
    tree_ids: &[i64],
    node_ids: &[i64],
    weights: &[f32],
    tree: i64,
    node: i64,
) -> f64 {
    tree_ids
        .iter()
        .zip(node_ids)
        .position(|(tree_id, node_id)| *tree_id == tree && *node_id == node)
        .and_then(|position| weights.get(position).copied())
        .map_or(0.0, f64::from)
}
