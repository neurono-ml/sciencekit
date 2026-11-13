//! Neighbour weighting: uniform or inverse-distance
//! (spec `nearest-neighbor-estimators`).

use sciencekit_common::SKFloat;

/// How a neighbour contributes to a vote or an average.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum SKNeighborsWeights {
    /// Every neighbour counts the same.
    #[default]
    Uniform,
    /// A neighbour counts by the inverse of its distance to the query row.
    ///
    /// When at least one selected neighbour sits exactly on the query row
    /// (distance zero), only those duplicates contribute and every farther
    /// neighbour is ignored. This keeps the result finite and predictable for
    /// duplicated training rows.
    Distance,
}

impl SKNeighborsWeights {
    /// The weight of each neighbour, in the given distance order.
    pub fn weights<F: SKFloat>(self, distances: &[F]) -> Vec<F> {
        let mut weights = vec![F::one(); distances.len()];
        if self == SKNeighborsWeights::Uniform {
            return weights;
        }
        let has_zero_distance = distances.iter().any(|distance| *distance == F::zero());
        for (weight, distance) in weights.iter_mut().zip(distances) {
            *weight = if has_zero_distance {
                if *distance == F::zero() {
                    F::one()
                } else {
                    F::zero()
                }
            } else {
                F::one() / *distance
            };
        }
        weights
    }
}

/// The weighted average of `values`, with the weights summing to the divisor.
pub fn sk_weighted_average<F: SKFloat>(weights: &[F], values: &[F]) -> F {
    let mut weighted_sum = F::zero();
    let mut total_weight = F::zero();
    for (weight, value) in weights.iter().zip(values) {
        weighted_sum = weighted_sum + *weight * *value;
        total_weight = total_weight + *weight;
    }
    weighted_sum / total_weight
}
