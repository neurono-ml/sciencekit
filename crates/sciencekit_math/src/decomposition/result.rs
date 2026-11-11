//! The result container of the randomized singular-value decomposition and the
//! deterministic sign convention applied to its singular vectors.

use ndarray::Array2;
use sciencekit_common::SKFloat;

/// The truncated singular-value decomposition produced by the randomized kernel.
///
/// The factorization is `X ≈ U Σ Vᵀ`, with `U` holding the `m × k` left
/// singular vectors, `V` the `n × k` right singular vectors and
/// `singular_values` the `k` non-increasing singular values.
#[derive(Debug, Clone)]
pub struct SKRandomizedSvdResult<F: SKFloat> {
    /// The left singular vectors, with shape `m × k`.
    pub u: Array2<F>,
    /// The singular values, non-increasing and non-negative.
    pub singular_values: Vec<F>,
    /// The right singular vectors, with shape `n × k`.
    pub v: Array2<F>,
}

impl<F: SKFloat> SKRandomizedSvdResult<F> {
    /// The number of retained components.
    pub fn number_of_components(&self) -> usize {
        self.singular_values.len()
    }

    /// Reconstructs the rank-`k` approximation `U Σ Vᵀ`.
    pub fn reconstruct(&self) -> Array2<F> {
        let components = self.number_of_components();
        let mut diagonal = Array2::<F>::zeros((components, components));
        for (index, &value) in self.singular_values.iter().enumerate() {
            diagonal[(index, index)] = value;
        }
        self.u.dot(&diagonal).dot(&self.v.t())
    }
}
