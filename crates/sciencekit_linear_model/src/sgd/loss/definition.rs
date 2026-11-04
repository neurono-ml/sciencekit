//! The sealed loss-function contract of the stochastic-gradient substrate
//! (spec `stochastic-gradient-solver`, decision 2).

use sciencekit_common::SKFloat;

/// A loss function consumed by the single-sample update.
///
/// It is **sealed**: only the catalogue in this module tree implements it, so a
/// regressor loss can never reach the classifier path even through a generic
/// bound. `prediction` is the linear predictor `xᵀw + b`; `target` is `±1` for
/// the classifier losses and a real value for the regressor losses.
pub trait SKSGDLoss<F: SKFloat>: private::SKSGDLossSealed<F> {
    /// The loss of one prediction against its target.
    fn value(&self, prediction: F, target: F) -> F;
    /// The subgradient of the loss with respect to the linear predictor.
    fn slope(&self, prediction: F, target: F) -> F;
    /// Whether the loss yields calibrated probabilities.
    fn is_probabilistic(&self) -> bool;
    /// The probability of the positive class for a linear predictor.
    ///
    /// Only meaningful when [`SKSGDLoss::is_probabilistic`] is true; the
    /// non-probabilistic losses return NaN and the model rejects the request
    /// before calling this method.
    fn probability(&self, prediction: F) -> F;
}

/// Private sealing module: the trait is public so the bound is nameable but
/// unreachable, which makes the seal exhaustive.
pub(crate) mod private {
    use sciencekit_common::SKFloat;

    /// The hidden supertrait whose only implementations live in this crate.
    pub trait SKSGDLossSealed<F: SKFloat> {}
}
