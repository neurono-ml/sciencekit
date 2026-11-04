//! Companion test for the regressor's epsilon-band validation (spec
//! `stochastic-gradient-regressor`).

use sciencekit_common::SKError;
use sciencekit_common::builders::SKBuilder;

use super::SKSGDRegressor;
use crate::sgd::SKSGDRegressorLoss;

#[test]
fn huber_rejects_a_zero_epsilon() {
    let error = SKSGDRegressor::<f64>::build()
        .loss(SKSGDRegressorLoss::Huber)
        .epsilon(0.0)
        .build()
        .unwrap_err();
    assert!(matches!(error, SKError::InvalidHyperparameter { name, .. } if name == "epsilon"));
    // Epsilon-insensitive losses keep accepting a zero band.
    for loss in [
        SKSGDRegressorLoss::SquaredError,
        SKSGDRegressorLoss::EpsilonInsensitive,
        SKSGDRegressorLoss::SquaredEpsilonInsensitive,
    ] {
        assert!(
            SKSGDRegressor::<f64>::build()
                .loss(loss)
                .epsilon(0.0)
                .build()
                .is_ok(),
            "the {loss:?} loss must accept a zero epsilon"
        );
    }
}
