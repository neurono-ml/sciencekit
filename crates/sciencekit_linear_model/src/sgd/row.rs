//! The representation-agnostic feature row the update consumes (spec
//! `stochastic-gradient-solver`, decision 9).
//!
//! A dense `ArrayView1` and a borrowed CSR outer row expose the same two
//! primitives, so the single-sample update never needs a dense copy of a sparse
//! design.

use ndarray::{ArrayView1, ArrayViewMut1, azip};
use sciencekit_common::SKFloat;
use sprs::CsVecView;

/// The per-row primitives of the single-sample update.
pub(crate) trait SKSGDFeatureRow<F: SKFloat> {
    /// The dot product of this row with the weights.
    fn dot_with(&self, weights: ArrayView1<F>) -> F;
    /// Add `scale · self` to the weights.
    fn add_scaled_to(&self, weights: ArrayViewMut1<F>, scale: F);
}

impl<F: SKFloat> SKSGDFeatureRow<F> for ArrayView1<'_, F> {
    fn dot_with(&self, weights: ArrayView1<F>) -> F {
        self.dot(&weights)
    }

    fn add_scaled_to(&self, mut weights: ArrayViewMut1<F>, scale: F) {
        azip!((weight in &mut weights, &feature in self) {
            *weight = *weight + scale * feature;
        });
    }
}

impl<F: SKFloat> SKSGDFeatureRow<F> for CsVecView<'_, F> {
    fn dot_with(&self, weights: ArrayView1<F>) -> F {
        self.indices()
            .iter()
            .zip(self.data())
            .map(|(&column, &value)| value * weights[column])
            .fold(F::zero(), |accumulator, term| accumulator + term)
    }

    fn add_scaled_to(&self, mut weights: ArrayViewMut1<F>, scale: F) {
        self.indices()
            .iter()
            .zip(self.data())
            .for_each(|(&column, &value)| {
                weights[column] = weights[column] + scale * value;
            });
    }
}
