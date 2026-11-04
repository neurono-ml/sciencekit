//! The running epoch objective (spec `stochastic-gradient-solver`, decision
//! 7).

use sciencekit_common::SKFloat;

/// The mean weighted loss accumulated over one pass of the data.
#[derive(Debug, Clone)]
pub(crate) struct SKEpochObjective<F> {
    total: F,
    count: usize,
}

impl<F: SKFloat> Default for SKEpochObjective<F> {
    fn default() -> Self {
        Self::new()
    }
}

impl<F: SKFloat> SKEpochObjective<F> {
    /// An empty objective.
    pub(crate) fn new() -> Self {
        Self {
            total: F::zero(),
            count: 0,
        }
    }

    /// Fold one sample's weighted loss into the mean.
    pub(crate) fn record(&mut self, weighted_loss: F) {
        self.total = self.total + weighted_loss;
        self.count += 1;
    }

    /// Fold another objective's samples into this one.
    pub(crate) fn merge(&mut self, other: &Self) {
        self.total = self.total + other.total;
        self.count += other.count;
    }

    /// The mean weighted loss, or zero before any sample was recorded.
    pub(crate) fn mean(&self) -> F {
        if self.count == 0 {
            F::zero()
        } else {
            self.total / F::from(self.count).expect("the count fits in the scalar")
        }
    }

    /// The number of samples recorded.
    #[cfg(test)]
    pub(crate) fn count(&self) -> usize {
        self.count
    }
}

/// Whether `current` improves on `previous` by more than `tolerance`.
pub(crate) fn sk_objective_improved<F: SKFloat>(previous: F, current: F, tolerance: F) -> bool {
    previous - current > tolerance
}
