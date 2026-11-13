//! Donor weighting schemes for the nearest-neighbor imputer
//! (spec `nearest-neighbor-imputation`).

use sciencekit_common::SKFloat;

/// How the observed values of the selected donors are averaged.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum SKImputationWeights {
    /// Every selected donor counts the same.
    #[default]
    Uniform,
    /// A donor counts by the inverse of its distance to the receiver row.
    ///
    /// When at least one selected donor sits exactly on the receiver row
    /// (distance zero), only those zero-distance donors contribute and every
    /// farther donor is ignored. This keeps the average finite for duplicated
    /// training rows.
    Distance,
}

impl SKImputationWeights {
    /// The weight of each selected donor, in the given distance order.
    pub fn weights<F: SKFloat>(self, distances: &[F]) -> Vec<F> {
        let mut weights = vec![F::one(); distances.len()];
        if self == SKImputationWeights::Uniform {
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
///
/// The caller guarantees at least one non-zero weight, so the divisor is never
/// zero.
pub fn sk_weighted_average<F: SKFloat>(weights: &[F], values: &[F]) -> F {
    let mut weighted_sum = F::zero();
    let mut total_weight = F::zero();
    for (weight, value) in weights.iter().zip(values) {
        weighted_sum = weighted_sum + *weight * *value;
        total_weight = total_weight + *weight;
    }
    weighted_sum / total_weight
}
