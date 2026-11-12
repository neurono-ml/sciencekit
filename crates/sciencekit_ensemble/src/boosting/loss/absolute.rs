//! Absolute-error loss `|y - f|` with constant unit Hessian surrogate.

use sciencekit_common::SKFloat;

use super::contract::{SKGradientLoss, sk_target_mean};

/// Absolute-error regression loss `L(f, y) = |y - f|`.
///
/// Because `∂²|y - f|/∂f² = 0` almost everywhere, `hessian` returns the
/// documented constant unit surrogate `1`, so the Newton leaf step averages the
/// signed residual signs rather than computing the leaf median.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct SKAbsoluteErrorLoss;

impl SKAbsoluteErrorLoss {
    /// Construct the absolute-error loss.
    pub fn new() -> Self {
        Self
    }
}

impl<F: SKFloat> SKGradientLoss<F> for SKAbsoluteErrorLoss {
    fn initial_prediction(&self, targets: &[F]) -> F {
        sk_target_mean(targets)
    }

    fn negative_gradient(&self, target: F, raw_prediction: F) -> F {
        if target > raw_prediction {
            F::one()
        } else if target < raw_prediction {
            -F::one()
        } else {
            F::zero()
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
        (target - raw_prediction).abs()
    }
}
