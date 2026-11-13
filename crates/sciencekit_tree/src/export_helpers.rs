//! Shared tree-state encoding for the decision trees and the forests.
//!
//! A tree travels as parallel node arrays: split features, thresholds, child
//! indices, sample counts and impurity per node, plus a flattened leaf-value
//! array with its per-node offsets. Children use `-1` and split features use
//! `-1` to mean "leaf"; thresholds use NaN at leaves.

use sciencekit_common::{
    SKError, SKFloat, SKModelState, SKTensor, sk_read_floating_tensor, sk_read_integer_tensor,
};

use crate::criterion::SKTreeCriterion;
use crate::node::{SKTree, SKTreeNode};

/// Encode a fitted tree under a name prefix.
pub fn export_tree<F: SKFloat>(
    tree: &SKTree<F>,
    prefix: &str,
) -> Result<Vec<SKTensor<F>>, SKError> {
    let nodes = tree.nodes();
    let split_features: Vec<i64> = nodes
        .iter()
        .map(|node| node.split_feature.map_or(-1, |feature| feature as i64))
        .collect();
    let thresholds: Vec<F> = nodes
        .iter()
        .map(|node| node.threshold.unwrap_or_else(F::zero))
        .collect();
    let left: Vec<i64> = nodes
        .iter()
        .map(|node| node.left_child.map_or(-1, |child| child as i64))
        .collect();
    let right: Vec<i64> = nodes
        .iter()
        .map(|node| node.right_child.map_or(-1, |child| child as i64))
        .collect();
    let sample_counts: Vec<i64> = nodes.iter().map(|node| node.sample_count as i64).collect();
    let impurities: Vec<F> = nodes.iter().map(|node| node.impurity).collect();
    let mut values = Vec::new();
    let mut value_offsets = Vec::with_capacity(nodes.len() + 1);
    value_offsets.push(0_i64);
    for node in nodes {
        values.extend(node.value.iter().copied());
        value_offsets.push(values.len() as i64);
    }
    Ok(vec![
        SKTensor::integer(
            format!("{prefix}.nodes.split_feature"),
            vec![nodes.len()],
            split_features,
        )?,
        SKTensor::floating(
            format!("{prefix}.nodes.threshold"),
            vec![nodes.len()],
            thresholds,
        )?,
        SKTensor::integer(format!("{prefix}.nodes.left"), vec![nodes.len()], left)?,
        SKTensor::integer(format!("{prefix}.nodes.right"), vec![nodes.len()], right)?,
        SKTensor::integer(
            format!("{prefix}.nodes.sample_count"),
            vec![nodes.len()],
            sample_counts,
        )?,
        SKTensor::floating(
            format!("{prefix}.nodes.impurity"),
            vec![nodes.len()],
            impurities,
        )?,
        SKTensor::floating(format!("{prefix}.nodes.value"), vec![values.len()], values)?,
        SKTensor::integer(
            format!("{prefix}.nodes.value_offsets"),
            vec![value_offsets.len()],
            value_offsets,
        )?,
    ])
}

/// Rebuild a fitted tree from a state.
pub fn import_tree<F: SKFloat>(
    state: &SKModelState<F>,
    prefix: &str,
    feature_count: usize,
) -> Result<SKTree<F>, SKError> {
    let split_features = sk_read_integer_tensor(state, &format!("{prefix}.nodes.split_feature"))?;
    let (_, thresholds) = sk_read_floating_tensor(state, &format!("{prefix}.nodes.threshold"))?;
    let left = sk_read_integer_tensor(state, &format!("{prefix}.nodes.left"))?;
    let right = sk_read_integer_tensor(state, &format!("{prefix}.nodes.right"))?;
    let sample_counts = sk_read_integer_tensor(state, &format!("{prefix}.nodes.sample_count"))?;
    let (_, impurities) = sk_read_floating_tensor(state, &format!("{prefix}.nodes.impurity"))?;
    let (_, values) = sk_read_floating_tensor(state, &format!("{prefix}.nodes.value"))?;
    let value_offsets = sk_read_integer_tensor(state, &format!("{prefix}.nodes.value_offsets"))?;
    let node_count = split_features.len();
    if thresholds.len() != node_count
        || left.len() != node_count
        || right.len() != node_count
        || sample_counts.len() != node_count
        || impurities.len() != node_count
        || value_offsets.len() != node_count + 1
    {
        return Err(SKError::ShapeMismatch {
            expected: vec![node_count, node_count, node_count],
            found: vec![
                thresholds.len(),
                left.len(),
                right.len(),
                sample_counts.len(),
                impurities.len(),
                value_offsets.len(),
            ],
        });
    }
    let mut nodes = Vec::with_capacity(node_count);
    for index in 0..node_count {
        let (split_feature, threshold, left_child, right_child) = (
            split_features[index],
            thresholds[index],
            left[index],
            right[index],
        );
        let start = value_offsets[index].max(0) as usize;
        let end = value_offsets[index + 1].max(0) as usize;
        let value = values
            .get(start..end)
            .ok_or(SKError::ShapeMismatch {
                expected: vec![values.len()],
                found: vec![end],
            })?
            .to_vec();
        let node = SKTreeNode {
            split_feature: (split_feature >= 0).then_some(split_feature as usize),
            threshold: (split_feature >= 0).then_some(threshold),
            left_child: (left_child >= 0).then_some(left_child as usize),
            right_child: (right_child >= 0).then_some(right_child as usize),
            sample_count: sample_counts[index].max(0) as usize,
            impurity: impurities[index],
            value,
        };
        if node
            .split_feature
            .is_some_and(|feature| feature >= feature_count)
        {
            return Err(SKError::ShapeMismatch {
                expected: vec![feature_count],
                found: vec![node.split_feature.unwrap_or_default()],
            });
        }
        nodes.push(node);
    }
    Ok(SKTree::new(nodes, feature_count))
}

/// The stable string for a split criterion.
pub fn criterion_name(criterion: SKTreeCriterion) -> &'static str {
    match criterion {
        SKTreeCriterion::Gini => "gini",
        SKTreeCriterion::Entropy => "entropy",
        SKTreeCriterion::SquaredError => "squared_error",
        SKTreeCriterion::AbsoluteError => "absolute_error",
    }
}

/// The split criterion for its stable string.
pub fn criterion_from_name(name: &str) -> SKTreeCriterion {
    match name {
        "entropy" => SKTreeCriterion::Entropy,
        "squared_error" => SKTreeCriterion::SquaredError,
        "absolute_error" => SKTreeCriterion::AbsoluteError,
        _ => SKTreeCriterion::Gini,
    }
}
