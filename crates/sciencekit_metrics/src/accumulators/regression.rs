//! Streaming regression accumulator.

use sciencekit_common::{SKError, SKFloat};

use crate::regression::SKCompensatedSum;

/// A mergeable accumulator of regression target/prediction pairs.
///
/// Holds five constant-size sums (count, `Σy`, `Σy²`, `Σ(y−ŷ)²`, `Σ|y−ŷ|`) in
/// compensated arithmetic, so batches and parallel partitions finalize to the
/// same metrics as a whole-input evaluation.
#[derive(Debug, Clone)]
pub struct SKRegressionAccumulator<F> {
    count: usize,
    targets: SKCompensatedSum<F>,
    squared_targets: SKCompensatedSum<F>,
    squared_errors: SKCompensatedSum<F>,
    absolute_errors: SKCompensatedSum<F>,
}

impl<F: SKFloat> SKRegressionAccumulator<F> {
    /// An empty accumulator.
    pub fn new() -> Self {
        Self {
            count: 0,
            targets: SKCompensatedSum::new(),
            squared_targets: SKCompensatedSum::new(),
            squared_errors: SKCompensatedSum::new(),
            absolute_errors: SKCompensatedSum::new(),
        }
    }

    /// Observe one target/prediction pair, rejecting non-finite values.
    pub fn observe(&mut self, target: F, prediction: F) -> Result<(), SKError> {
        if !target.is_finite() || !prediction.is_finite() {
            return Err(SKError::NonFiniteValue {
                position: self.count,
            });
        }
        let error = target - prediction;
        self.targets.add(target);
        self.squared_targets.add(target * target);
        self.squared_errors.add(error * error);
        self.absolute_errors.add(error.abs());
        self.count += 1;
        Ok(())
    }

    /// Merge another accumulator into this one.
    pub fn merge(&mut self, other: &Self) {
        self.targets.merge(other.targets);
        self.squared_targets.merge(other.squared_targets);
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
        let mean = self.targets.value() / self.count_scalar();
        let total = self.squared_targets.value() - self.targets.value() * mean;
        let residual = self.squared_errors.value();
        if total <= F::zero() {
            return Ok(if residual.is_zero() {
                F::one()
            } else {
                F::zero()
            });
        }
        Ok(F::one() - residual / total)
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
