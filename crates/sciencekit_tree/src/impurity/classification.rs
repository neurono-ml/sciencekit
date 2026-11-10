//! Classification impurity over class-count vectors.

use sciencekit_common::SKFloat;

/// Gini impurity `1 - Σ pᵢ²` over a vector of class counts or weights.
///
/// Returns zero for an empty or all-zero vector. The ideal, pure node has
/// impurity zero; the impurity is bounded above by `1 - 1 / k` for `k` classes.
pub fn sk_gini_impurity<F: SKFloat>(class_counts: &[F]) -> F {
    let total = class_counts
        .iter()
        .fold(F::zero(), |sum, count| sum + *count);
    if total <= F::zero() {
        return F::zero();
    }
    let sum_of_squares = class_counts.iter().fold(F::zero(), |sum, count| {
        let probability = *count / total;
        sum + probability * probability
    });
    F::one() - sum_of_squares
}

/// Shannon entropy `-Σ pᵢ log pᵢ` over a vector of class counts or weights.
///
/// Uses the natural logarithm and returns zero for an empty or all-zero
/// vector; a zero-probability class contributes zero.
pub fn sk_entropy_impurity<F: SKFloat>(class_counts: &[F]) -> F {
    let total = class_counts
        .iter()
        .fold(F::zero(), |sum, count| sum + *count);
    if total <= F::zero() {
        return F::zero();
    }
    let entropy = class_counts.iter().fold(F::zero(), |sum, count| {
        let probability = *count / total;
        if probability > F::zero() {
            sum - probability * probability.ln()
        } else {
            sum
        }
    });
    entropy
}
