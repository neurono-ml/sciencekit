//! The regressor loss adapter and the schedule's initial slope scale (spec
//! `stochastic-gradient-regressor`).

use sciencekit_common::SKFloat;

use crate::sgd::{
    SKSGDLoss, SKSGDRegressorLoss, SKSGDRegressorLossAdapter, sk_regressor_loss_adapter,
};

/// Bind the selected regression loss to its `epsilon` band.
///
/// `epsilon` is consulted only by `Huber`, `EpsilonInsensitive` and
/// `SquaredEpsilonInsensitive`; binding it for `SquaredError` is harmless
/// because that loss never reads the band.
pub(crate) fn sk_loss_adapter<F: SKFloat>(
    loss: SKSGDRegressorLoss,
    epsilon: F,
) -> SKSGDRegressorLossAdapter<F> {
    sk_regressor_loss_adapter(loss, epsilon)
}

/// The mean absolute slope at a zero predictor, used to estimate the `Optimal`
/// schedule's fixed offset `t₀`.
pub(crate) fn sk_initial_slope_scale<F: SKFloat>(
    loss: SKSGDRegressorLoss,
    epsilon: F,
    targets: &[F],
) -> F {
    if targets.is_empty() {
        return F::one();
    }
    let adapter = sk_loss_adapter(loss, epsilon);
    let total = targets.iter().fold(F::zero(), |accumulator, &target| {
        accumulator + adapter.slope(F::zero(), target).abs()
    });
    total / F::from(targets.len()).expect("the target count fits in the scalar")
}
