//! Recursive farthest-pivot node construction for the ball tree.

use ndarray::Array2;
use sciencekit_common::SKFloat;

use super::super::node::SKBallTreeNode;

/// Build the subtree over `[start, end)`, returning its node index and depth.
pub(super) fn build_node<F: SKFloat>(
    features: &Array2<F>,
    permutation: &mut [usize],
    start: usize,
    end: usize,
    leaf_size: usize,
    nodes: &mut Vec<SKBallTreeNode<F>>,
) -> (usize, usize) {
    let centroid = centroid_of(features, permutation, start, end);
    let radius = permutation[start..end]
        .iter()
        .fold(F::zero(), |maximum, &row| {
            maximum.max(distance(features, row, &centroid))
        });
    let node_index = nodes.len();
    nodes.push(SKBallTreeNode::new(start, end, centroid, radius));

    if end - start <= leaf_size {
        return (node_index, 1);
    }

    let centroid = nodes[node_index].centroid().to_vec();
    let pivot_left = farthest_row(features, permutation, start, end, &centroid, None);
    let pivot_right = farthest_row(
        features,
        permutation,
        start,
        end,
        &features.row(pivot_left).to_vec(),
        Some(pivot_left),
    );
    let middle = start + (end - start) / 2;
    permutation[start..end].select_nth_unstable_by(middle - start, |&left, &right| {
        let left_gap = squared_distance(features, left, pivot_left);
        let right_gap = squared_distance(features, left, pivot_right);
        let left_other = squared_distance(features, right, pivot_left);
        let right_other = squared_distance(features, right, pivot_right);
        (left_gap - right_gap)
            .partial_cmp(&(left_other - right_other))
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| left.cmp(&right))
    });

    let (left, left_depth) = build_node(features, permutation, start, middle, leaf_size, nodes);
    let (right, right_depth) = build_node(features, permutation, middle, end, leaf_size, nodes);
    nodes[node_index].left = Some(left);
    nodes[node_index].right = Some(right);
    (node_index, 1 + left_depth.max(right_depth))
}

fn centroid_of<F: SKFloat>(
    features: &Array2<F>,
    permutation: &[usize],
    start: usize,
    end: usize,
) -> Vec<F> {
    let features_count = features.ncols();
    let mut centroid = vec![F::zero(); features_count];
    for &row in &permutation[start..end] {
        for (column, &value) in features.row(row).iter().enumerate() {
            centroid[column] = centroid[column] + value;
        }
    }
    let count = F::from(end - start).expect("row count fits in the scalar");
    centroid.iter().map(|&total| total / count).collect()
}

fn farthest_row<F: SKFloat>(
    features: &Array2<F>,
    permutation: &[usize],
    start: usize,
    end: usize,
    target: &[F],
    excluded: Option<usize>,
) -> usize {
    let mut best_row = permutation[start];
    let mut best_distance = F::neg_infinity();
    for &row in &permutation[start..end] {
        if Some(row) == excluded {
            continue;
        }
        let candidate = distance(features, row, target);
        if candidate > best_distance {
            best_distance = candidate;
            best_row = row;
        }
    }
    best_row
}

fn distance<F: SKFloat>(features: &Array2<F>, row: usize, target: &[F]) -> F {
    features
        .row(row)
        .iter()
        .zip(target.iter())
        .fold(F::zero(), |total, (&value, &other)| {
            let difference = value - other;
            total + difference * difference
        })
        .sqrt()
}

fn squared_distance<F: SKFloat>(features: &Array2<F>, left: usize, right: usize) -> F {
    features
        .row(left)
        .iter()
        .zip(features.row(right).iter())
        .fold(F::zero(), |total, (&a, &b)| {
            let difference = a - b;
            total + difference * difference
        })
}
