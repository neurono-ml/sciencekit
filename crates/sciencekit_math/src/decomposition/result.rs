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

/// Fixes the sign of every singular-vector pair.
///
/// A singular vector is defined only up to sign, so the kernel makes the
/// largest-magnitude entry of each left singular vector positive. An exact tie
/// is resolved by the smallest row index. The matching column of the right
/// singular vectors is flipped with it, so `U Σ Vᵀ` is preserved.
pub(crate) fn sk_apply_sign_convention<F: SKFloat>(u: &mut Array2<F>, v: &mut Array2<F>) {
    let components = u.ncols();
    for component in 0..components {
        let mut pivot_row = 0usize;
        let mut pivot_magnitude = F::zero();
        for row in 0..u.nrows() {
            let magnitude = u[(row, component)].abs();
            if row == 0 || magnitude > pivot_magnitude {
                pivot_magnitude = magnitude;
                pivot_row = row;
            }
        }
        if u[(pivot_row, component)] < F::zero() {
            for row in 0..u.nrows() {
                u[(row, component)] = -u[(row, component)];
            }
            if component < v.ncols() {
                for row in 0..v.nrows() {
                    v[(row, component)] = -v[(row, component)];
                }
            }
        }
    }
}
