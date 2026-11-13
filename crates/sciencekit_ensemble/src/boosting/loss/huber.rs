//! Huber regression loss: quadratic within `[-delta, delta]`, linear beyond,
//! with clipped negative gradient and constant unit Hessian surrogate.

use sciencekit_common::{SKError, SKFloat, sk_validate_hyperparameter};

use super::contract::{SKGradientLoss, sk_target_mean};

/// Huber regression loss parameterized by positive clipping threshold `delta`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SKHuberLoss<F> {
    delta: F,
}

impl<F: SKFloat> SKHuberLoss<F> {
    /// Start building a Huber loss (default `delta = 1.0`).
    pub fn build() -> SKHuberLossBuilder<F> {
        SKHuberLossBuilder::new()
    }

    /// The configured gradient-clipping threshold `delta`.
    pub fn delta(&self) -> F {
        self.delta
    }
}

/// Builder for [`SKHuberLoss`].
#[derive(Debug, Clone, Copy)]
pub struct SKHuberLossBuilder<F> {
    delta: F,
}

impl<F: SKFloat> SKHuberLossBuilder<F> {
    /// Create a builder with default `delta = 1.0`.
    pub fn new() -> Self {
        Self { delta: F::one() }
    }

    /// Configure the positive clipping threshold `delta` (default `1.0`).
    pub fn delta(mut self, delta: F) -> Self {
        self.delta = delta;
        self
    }

    /// Validate `delta` and construct [`SKHuberLoss`].
    pub fn build(self) -> Result<SKHuberLoss<F>, SKError> {
        sk_validate_hyperparameter(
            "delta",
            self.delta.is_finite() && self.delta > F::zero(),
            "Huber delta must be positive and finite",
        )?;
        Ok(SKHuberLoss { delta: self.delta })
    }
}

impl<F: SKFloat> Default for SKHuberLossBuilder<F> {
    fn default() -> Self {
        Self::new()
    }
}

impl<F: SKFloat> SKGradientLoss<F> for SKHuberLoss<F> {
    fn initial_prediction(&self, targets: &[F]) -> F {
        sk_target_mean(targets)
    }

    fn negative_gradient(&self, target: F, raw_prediction: F) -> F {
        let residual = target - raw_prediction;
        if residual > self.delta {
            self.delta
        } else if residual < -self.delta {
            -self.delta
        } else {
            residual
        }
    }

    fn hessian(&self, _target: F, _raw_prediction: F) -> F {
        F::one()
    }

    fn link(&self, prediction: F) -> F {
        prediction
    }

    fn inverse_link(&self, raw_prediction: F) -> F {
        raw_prediction
    }

    fn loss(&self, raw_prediction: F, target: F) -> F {
        let absolute_residual = (target - raw_prediction).abs();
        let half = F::from(0.5).expect("0.5 fits in scalar");
        if absolute_residual <= self.delta {
            half * absolute_residual * absolute_residual
        } else {
            self.delta * (absolute_residual - half * self.delta)
        }
    }
}
