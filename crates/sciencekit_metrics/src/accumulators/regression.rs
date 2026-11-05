//! Streaming regression accumulator.

use sciencekit_common::{SKError, SKFloat};

use crate::regression::SKCompensatedSum;

/// A mergeable accumulator of regression target/prediction pairs.
///
/// Holds the observation count, the running mean and `M2` (the sum of squared
/// deviations from the mean, in the Welford/Chan form) plus compensated error
/// sums. The `M2` state avoids the catastrophic cancellation of `Σy² − n·mean²`
/// for large-magnitude `f32` targets, so batches and parallel partitions
/// finalize to the same metrics as a whole-input evaluation.
#[derive(Debug, Clone)]
pub struct SKRegressionAccumulator<F> {
    count: usize,
    mean: F,
    m2: F,
    squared_errors: SKCompensatedSum<F>,
    absolute_errors: SKCompensatedSum<F>,
}

impl<F: SKFloat> SKRegressionAccumulator<F> {
    /// An empty accumulator.
    pub fn new() -> Self {
        Self {
            count: 0,
            mean: F::zero(),
            m2: F::zero(),
            squared_errors: SKCompensatedSum::new(),
            absolute_errors: SKCompensatedSum::new(),
        }
    }

    /// Observe one target/prediction pair, rejecting non-finite values.
    ///
    /// The reported position is the observation's index in this accumulator's
    /// stream, not its index inside the current batch.
    pub fn observe(&mut self, target: F, prediction: F) -> Result<(), SKError> {
        if !target.is_finite() || !prediction.is_finite() {
            return Err(SKError::NonFiniteValue {
                position: self.count,
            });
        }
        let error = target - prediction;
        self.count += 1;
        let delta = target - self.mean;
        self.mean = self.mean + delta / self.count_scalar();
        let delta_after = target - self.mean;
        self.m2 = self.m2 + delta * delta_after;
        self.squared_errors.add(error * error);
        self.absolute_errors.add(error.abs());
        Ok(())
    }

    /// Merge another accumulator into this one.
    pub fn merge(&mut self, other: &Self) {
        if other.count == 0 {
            return;
        }
        if self.count == 0 {
            *self = other.clone();
            return;
        }
        let left_count = self.count_scalar();
        let right_count = other.count_scalar();
        let total = left_count + right_count;
        let delta = other.mean - self.mean;
        self.mean = self.mean + delta * right_count / total;
        self.m2 = self.m2 + other.m2 + delta * delta * left_count * right_count / total;
        self.squared_errors.merge(other.squared_errors);
        self.absolute_errors.merge(other.absolute_errors);
        self.count += other.count;
    }

    /// The number of observations accumulated.
    pub fn number_of_observations(&self) -> usize {
        self.count
    }

    /// The mean squared error over the accumulated pairs.
    pub fn mean_squared_error(&self) -> Result<F, SKError> {
        self.require_observations("mean_squared_error")?;
        Ok(self.squared_errors.value() / self.count_scalar())
    }

    /// The root mean squared error over the accumulated pairs.
    pub fn root_mean_squared_error(&self) -> Result<F, SKError> {
        self.require_observations("root_mean_squared_error")?;
        Ok((self.squared_errors.value() / self.count_scalar()).sqrt())
    }

    /// The mean absolute error over the accumulated pairs.
    pub fn mean_absolute_error(&self) -> Result<F, SKError> {
        self.require_observations("mean_absolute_error")?;
        Ok(self.absolute_errors.value() / self.count_scalar())
    }

    /// The coefficient of determination over the accumulated pairs.
    pub fn r2_score(&self) -> Result<F, SKError> {
        self.require_observations("r2")?;
        let total = self.m2;
        if total <= F::zero() {
            return Ok(if self.squared_errors.value().is_zero() {
                F::one()
            } else {
                F::zero()
            });
        }
        Ok(F::one() - self.squared_errors.value() / total)
    }

    /// Reject finalization over no observations.
    fn require_observations(&self, operation: &'static str) -> Result<(), SKError> {
        if self.count == 0 {
            return Err(SKError::EmptyInput { operation });
        }
        Ok(())
    }

    /// The observation count in the scalar.
    fn count_scalar(&self) -> F {
        num_traits::cast::<usize, F>(self.count).expect("counts fit in the scalar")
    }
}

impl<F: SKFloat> Default for SKRegressionAccumulator<F> {
    fn default() -> Self {
        Self::new()
    }
}
