//! The scalar loss/gradient/Hessian contract consumed by boosting stages.

use sciencekit_common::SKFloat;

/// Differentiable scalar loss contract shared across regression and binary
/// classification boosting loops.
pub trait SKGradientLoss<F: SKFloat>: Send + Sync {
    /// Constant raw-score initialization from the observed training targets.
    fn initial_prediction(&self, targets: &[F]) -> F;

    /// Negative first derivative `-∂L(y, f)/∂f` at `raw_prediction`.
    fn negative_gradient(&self, target: F, raw_prediction: F) -> F;

    /// Second derivative `∂²L(y, f)/∂f²` (or documented positive surrogate) at
    /// `raw_prediction`.
    fn hessian(&self, target: F, raw_prediction: F) -> F;

    /// Map a prediction-space value into the additive raw-score space.
    fn link(&self, prediction: F) -> F;

    /// Map an additive raw score into the observable prediction space.
    fn inverse_link(&self, raw_prediction: F) -> F;

    /// Pointwise loss `L(raw_prediction, target)` used for validation scoring.
    fn loss(&self, raw_prediction: F, target: F) -> F;
}

/// Arithmetic mean of a target slice, or zero when empty.
pub(super) fn sk_target_mean<F: SKFloat>(targets: &[F]) -> F {
    if targets.is_empty() {
        return F::zero();
    }
    let sum = targets
        .iter()
        .fold(F::zero(), |accumulator, &value| accumulator + value);
    sum / F::from(targets.len()).expect("target count fits in scalar")
}
