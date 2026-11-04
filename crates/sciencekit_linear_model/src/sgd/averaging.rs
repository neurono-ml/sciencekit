//! The running averaged iterate (ASGD) (spec `stochastic-gradient-solver`,
//! decision 5).

use ndarray::{Array1, ArrayView1, azip};
use sciencekit_common::SKFloat;

/// The running mean of the iterates visited by the driver.
///
/// Updated in place with `w̄ ← w̄ + (w − w̄) / t`, so the trajectory is never
/// stored: one vector plus the intercept is the whole state, and the counter
/// survives a `partial_fit` continuation.
#[derive(Debug, Clone)]
pub(crate) struct SKAveragedIterate<F> {
    weights: Array1<F>,
    intercept: F,
    count: usize,
}

impl<F: SKFloat> SKAveragedIterate<F> {
    /// Start the average at the initial iterate.
    pub(crate) fn new(weights: Array1<F>, intercept: F) -> Self {
        Self {
            weights,
            intercept,
            count: 0,
        }
    }

    /// Restore an average from its persisted parts, continuing its counter.
    #[cfg(test)]
    pub(crate) fn from_parts(weights: Array1<F>, intercept: F, count: usize) -> Self {
        Self {
            weights,
            intercept,
            count,
        }
    }

    /// Fold one visited iterate into the running mean.
    pub(crate) fn record(&mut self, weights: ArrayView1<F>, intercept: F) {
        self.count += 1;
        let inverse_count = F::one() / F::from(self.count).expect("the count fits in the scalar");
        azip!((average in &mut self.weights, &value in &weights) {
            *average = *average + (value - *average) * inverse_count;
        });
        self.intercept = self.intercept + (intercept - self.intercept) * inverse_count;
    }

    /// Restart the average from a chosen iterate (used by early stopping so a
    /// continuation starts from the iterate the model reports).
    pub(crate) fn reset(&mut self, weights: ArrayView1<F>, intercept: F) {
        self.weights.assign(&weights);
        self.intercept = intercept;
        self.count = 0;
    }

    /// The averaged coefficients.
    pub(crate) fn weights(&self) -> ArrayView1<F> {
        self.weights.view()
    }

    /// The averaged intercept.
    pub(crate) fn intercept(&self) -> F {
        self.intercept
    }

    /// The number of iterates folded into the mean.
    #[cfg(test)]
    pub(crate) fn count(&self) -> usize {
        self.count
    }
}
