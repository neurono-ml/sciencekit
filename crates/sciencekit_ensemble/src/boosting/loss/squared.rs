//! Half squared-error loss `0.5 * (y - f)²` for regression boosting.

use sciencekit_common::SKFloat;

use super::contract::{SKGradientLoss, sk_target_mean};

/// Half squared-error regression loss `L(f, y) = 0.5 * (y - f)²`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct SKSquaredErrorLoss;

impl SKSquaredErrorLoss {
    /// Construct the squared-error loss.
    pub fn new() -> Self {
        Self
    }
}

impl<F: SKFloat> SKGradientLoss<F> for SKSquaredErrorLoss {
    fn initial_prediction(&self, targets: &[F]) -> F {
        sk_target_mean(targets)
    }

    fn negative_gradient(&self, target: F, raw_prediction: F) -> F {
        target - raw_prediction
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
        let residual = target - raw_prediction;
        let half = F::from(0.5).expect("0.5 fits in scalar");
        half * residual * residual
    }
}
