//! Weighted class scores and deterministic label selection
//! (spec `nearest-neighbor-estimators`).

use ndarray::ArrayView1;
use sciencekit_common::SKFloat;

use crate::neighbors::SKNeighborsWeights;

/// The weighted vote score of every class for one query row.
///
/// The returned vector has one entry per canonical class, including classes
/// absent from the neighbourhood (their score is zero).
pub(crate) fn sk_class_scores<F: SKFloat>(
    distances: &[F],
    neighbor_indices: &[usize],
    class_indices: ArrayView1<'_, usize>,
    number_of_classes: usize,
    weights: SKNeighborsWeights,
) -> Vec<F> {
    let neighbor_weights = weights.weights(distances);
    let mut scores = vec![F::zero(); number_of_classes];
    for (weight, index) in neighbor_weights.iter().zip(neighbor_indices) {
        scores[class_indices[*index]] = scores[class_indices[*index]] + *weight;
    }
    scores
}

/// The class with the largest score, ties broken by the smallest canonical
/// index.
pub(crate) fn sk_argmax_label<F: SKFloat>(scores: &[F]) -> i64 {
    let mut best = 0;
    for (index, score) in scores.iter().enumerate() {
        if *score > scores[best] {
            best = index;
        }
    }
    best as i64
}

/// Normalize class scores into probabilities; a zero total stays all zero.
pub(crate) fn sk_normalize_scores<F: SKFloat>(scores: &[F]) -> Vec<F> {
    let total = scores.iter().fold(F::zero(), |sum, score| sum + *score);
    if total > F::zero() {
        scores.iter().map(|score| *score / total).collect()
    } else {
        scores.to_vec()
    }
}
