//! Neumaier compensated summation for regression metrics.

use sciencekit_common::SKFloat;

/// A Neumaier compensated running sum.
///
/// Keeps a compensation term for the low-order bits lost when a small addend
/// meets a large accumulator, so long regression sums stay accurate in `f32`
/// as well as `f64`.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct SKCompensatedSum<F> {
    sum: F,
    compensation: F,
}

impl<F: SKFloat> SKCompensatedSum<F> {
    /// An empty sum.
    pub(crate) fn new() -> Self {
        Self {
            sum: F::zero(),
            compensation: F::zero(),
        }
    }

    /// Add one value.
    pub(crate) fn add(&mut self, value: F) {
        let updated = self.sum + value;
        if self.sum.abs() >= value.abs() {
            self.compensation = self.compensation + ((self.sum - updated) + value);
        } else {
            self.compensation = self.compensation + ((value - updated) + self.sum);
        }
        self.sum = updated;
    }

    /// The sum including the compensation term.
    pub(crate) fn value(&self) -> F {
        self.sum + self.compensation
    }

    /// Merge another partial sum into this one.
    pub(crate) fn merge(&mut self, other: Self) {
        self.add(other.sum);
        self.add(other.compensation);
    }
}
