//! Random isolation tree growth over the shared `SKTree` node array.

#![cfg_attr(not(test), allow(dead_code))]

use ndarray::ArrayView2;
use sciencekit_common::{SKFloat, SKRandomGenerator};
use sciencekit_tree::{SKTree, SKTreeNode};

use super::subsample::{DenseSubsample, IsolationSampleSource};

/// Computes `ceil(log2(sample_size))` for `sample_size > 1`, or `0` otherwise.
pub(crate) fn sk_isolation_depth_limit(sample_size: usize) -> usize {
    if sample_size <= 1 {
        0
    } else {
        (sample_size - 1).ilog2() as usize + 1
    }
}

/// Builds a single isolation tree from a dense subsample matrix.
pub(crate) fn sk_build_dense_isolation_tree<F: SKFloat>(
    subsample: ArrayView2<'_, F>,
    candidate_features: &[usize],
    depth_limit: usize,
    generator: &mut SKRandomGenerator,
) -> SKTree<F> {
    let source = DenseSubsample::new(subsample);
    sk_grow_isolation_tree(&source, candidate_features, depth_limit, generator)
}

/// Grows an isolation tree from any [`IsolationSampleSource`].
pub(crate) fn sk_grow_isolation_tree<F: SKFloat, S: IsolationSampleSource<F>>(
    source: &S,
    candidate_features: &[usize],
    depth_limit: usize,
    generator: &mut SKRandomGenerator,
) -> SKTree<F> {
    let sample_count = source.sample_count();
    if sample_count == 0 {
        return SKTree::new(
            vec![SKTreeNode::leaf(0, F::zero(), Vec::new())],
            source.feature_count(),
        );
    }
    let initial_rows: Vec<usize> = (0..sample_count).collect();
    let mut nodes = Vec::with_capacity(sample_count.saturating_mul(2).max(1));
    grow_node(
        source,
        candidate_features,
        depth_limit,
        0,
        initial_rows,
        generator,
        &mut nodes,
    );
    SKTree::new(nodes, source.feature_count())
}

fn grow_node<F: SKFloat, S: IsolationSampleSource<F>>(
    source: &S,
    candidate_features: &[usize],
    depth_limit: usize,
    current_depth: usize,
    rows: Vec<usize>,
    generator: &mut SKRandomGenerator,
    nodes: &mut Vec<SKTreeNode<F>>,
) -> usize {
    let node_sample_count = rows.len();
    if node_sample_count <= 1 || current_depth >= depth_limit || candidate_features.is_empty() {
        let leaf_index = nodes.len();
        nodes.push(SKTreeNode::leaf(node_sample_count, F::zero(), Vec::new()));
        return leaf_index;
    }

    let Some((selected_feature, minimum_value, maximum_value)) =
        select_varying_feature(source, &rows, candidate_features, generator)
    else {
        let leaf_index = nodes.len();
        nodes.push(SKTreeNode::leaf(node_sample_count, F::zero(), Vec::new()));
        return leaf_index;
    };

    let threshold = sample_split_threshold(minimum_value, maximum_value, generator);
    let (left_rows, right_rows) = source.partition_rows(&rows, selected_feature, threshold);

    let node_index = nodes.len();
    nodes.push(SKTreeNode::leaf(node_sample_count, F::zero(), Vec::new()));
    let left_child = grow_node(
        source,
        candidate_features,
        depth_limit,
        current_depth + 1,
        left_rows,
        generator,
        nodes,
    );
    let right_child = grow_node(
        source,
        candidate_features,
        depth_limit,
        current_depth + 1,
        right_rows,
        generator,
        nodes,
    );
    nodes[node_index] = SKTreeNode::split(
        selected_feature,
        threshold,
        left_child,
        right_child,
        node_sample_count,
        F::zero(),
    );
    node_index
}

fn select_varying_feature<F: SKFloat, S: IsolationSampleSource<F>>(
    source: &S,
    rows: &[usize],
    candidate_features: &[usize],
    generator: &mut SKRandomGenerator,
) -> Option<(usize, F, F)> {
    let first_position = generator.next_index(candidate_features.len());
    let first_feature = candidate_features[first_position];
    let (first_minimum, first_maximum) = source.feature_bounds(rows, first_feature);
    if first_minimum < first_maximum {
        return Some((first_feature, first_minimum, first_maximum));
    }
    if candidate_features.len() == 1 {
        return None;
    }
    let mut remaining: Vec<usize> = candidate_features
        .iter()
        .enumerate()
        .filter_map(|(position, &feature)| (position != first_position).then_some(feature))
        .collect();
    generator.shuffle(&mut remaining);
    remaining.into_iter().find_map(|feature| {
        let (minimum_value, maximum_value) = source.feature_bounds(rows, feature);
        (minimum_value < maximum_value).then_some((feature, minimum_value, maximum_value))
    })
}

fn sample_split_threshold<F: SKFloat>(
    minimum_value: F,
    maximum_value: F,
    generator: &mut SKRandomGenerator,
) -> F {
    let uniform_draw = generator.next_f64();
    let minimum_f64 = minimum_value.to_f64().expect("finite scalar fits f64");
    let maximum_f64 = maximum_value.to_f64().expect("finite scalar fits f64");
    let interpolated = (1.0 - uniform_draw) * minimum_f64 + uniform_draw * maximum_f64;
    let mut threshold = F::from(interpolated).unwrap_or(minimum_value);
    if threshold < minimum_value {
        threshold = minimum_value;
    }
    if threshold >= maximum_value {
        let midpoint = F::from(0.5 * minimum_f64 + 0.5 * maximum_f64).unwrap_or(minimum_value);
        threshold = if midpoint >= minimum_value && midpoint < maximum_value {
            midpoint
        } else {
            minimum_value
        };
    }
    threshold
}
