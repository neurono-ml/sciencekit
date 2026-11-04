//! The regressor loss catalogue (spec `stochastic-gradient-solver`,
//! decision 2).
//!
//! Targets are real values. The insensitive losses carry the `epsilon` band,
//! so the catalogue is adapted into a loss value through
//! [`sk_regressor_loss_adapter`].

use sciencekit_common::SKFloat;

use super::definition::SKSGDLoss;
use super::definition::private::SKSGDLossSealed;

/// The loss of a linear regressor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SKSGDRegressorLoss {
    /// The squared error `0.5·(z − y)²`.
    SquaredError,
    /// The Huber loss, quadratic inside the `epsilon` band and linear beyond.
    Huber,
    /// The epsilon-insensitive loss `max(0, |z − y| − epsilon)`.
    EpsilonInsensitive,
    /// The squared epsilon-insensitive loss.
    SquaredEpsilonInsensitive,
}

/// A regressor loss bound to its `epsilon` band.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SKSGDRegressorLossAdapter<F> {
    loss: SKSGDRegressorLoss,
    epsilon: F,
}

/// Bind a regressor loss to its `epsilon` band.
pub(crate) fn sk_regressor_loss_adapter<F: SKFloat>(
    loss: SKSGDRegressorLoss,
    epsilon: F,
) -> SKSGDRegressorLossAdapter<F> {
    SKSGDRegressorLossAdapter { loss, epsilon }
}

impl<F: SKFloat> SKSGDLossSealed<F> for SKSGDRegressorLossAdapter<F> {}

impl<F: SKFloat> SKSGDLoss<F> for SKSGDRegressorLossAdapter<F> {
    fn value(&self, prediction: F, target: F) -> F {
        let residual = prediction - target;
        match self.loss {
            SKSGDRegressorLoss::SquaredError => {
                F::from(0.5).expect("one half is representable") * residual * residual
            }
            SKSGDRegressorLoss::Huber => {
                let absolute = residual.abs();
                if absolute <= self.epsilon {
                    F::from(0.5).expect("one half is representable") * residual * residual
                } else {
                    self.epsilon * (absolute - F::from(0.5).expect("one half") * self.epsilon)
                }
            }
            SKSGDRegressorLoss::EpsilonInsensitive => {
                (residual.abs() - self.epsilon).max(F::zero())
            }
            SKSGDRegressorLoss::SquaredEpsilonInsensitive => {
                let excess = (residual.abs() - self.epsilon).max(F::zero());
                F::from(0.5).expect("one half is representable") * excess * excess
            }
        }
    }

    fn slope(&self, prediction: F, target: F) -> F {
        let residual = prediction - target;
        match self.loss {
            SKSGDRegressorLoss::SquaredError => residual,
            SKSGDRegressorLoss::Huber => {
                if residual.abs() <= self.epsilon {
                    residual
                } else {
                    self.epsilon * residual.signum()
                }
            }
            SKSGDRegressorLoss::EpsilonInsensitive => {
                if residual.abs() > self.epsilon {
                    residual.signum()
                } else {
                    F::zero()
                }
            }
            SKSGDRegressorLoss::SquaredEpsilonInsensitive => {
                if residual.abs() > self.epsilon {
                    residual.signum() * (residual.abs() - self.epsilon)
                } else {
                    F::zero()
                }
            }
        }
    }

    fn is_probabilistic(&self) -> bool {
        false
    }

    fn probability(&self, _prediction: F) -> F {
        F::nan()
    }
}

impl<F: SKFloat> SKSGDRegressorLossAdapter<F> {
    /// The epsilon band the loss was bound to.
    pub(crate) fn epsilon(&self) -> F {
        self.epsilon
    }
}
