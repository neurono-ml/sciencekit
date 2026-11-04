//! Construction and validation of the SGD regressor builder (spec
//! `stochastic-gradient-regressor`).

use sciencekit_common::{
    SKBackendKind, SKBuilderState, SKError, SKExecutionMode, SKFloat,
    builders::{SKBuilder, sk_validate_hyperparameter},
};

use super::super::coefficients::SKSGDRegressorCoefficients;
use super::super::core_implementation::SKSGDRegressor;
use crate::sgd::config::{SKSGDConfig, sk_validate_sgd_configuration};
use crate::sgd::{SKSGDLearningRate, SKSGDRegressorLoss};

/// Builder for [`SKSGDRegressor`]: the only construction surface.
pub struct SKSGDRegressorBuilder<F> {
    pub(super) state: SKBuilderState,
    pub(super) config: SKSGDConfig<F>,
    pub(super) loss: SKSGDRegressorLoss,
    pub(super) epsilon: F,
    pub(super) warm_start: Option<SKSGDRegressorCoefficients<F>>,
    pub(super) backend: SKBackendKind,
}

impl<F: SKFloat> SKSGDRegressorBuilder<F> {
    /// A fresh builder with the documented defaults.
    pub(crate) fn new() -> SKSGDRegressorBuilder<F> {
        Self {
            state: SKBuilderState::new(),
            config: SKSGDConfig::<F> {
                learning_rate: SKSGDLearningRate::InverseScaling,
                ..SKSGDConfig::default()
            },
            loss: SKSGDRegressorLoss::SquaredError,
            epsilon: F::from(0.1).expect("the default epsilon is representable"),
            warm_start: None,
            backend: SKBackendKind::Faer,
        }
    }
}

impl<F: SKFloat> Default for SKSGDRegressorBuilder<F> {
    fn default() -> Self {
        Self::new()
    }
}

impl<F: SKFloat> SKBuilder<SKSGDRegressor<F>> for SKSGDRegressorBuilder<F> {
    fn execution_mode<E>(&mut self, mode: E) -> &mut Self
    where
        E: TryInto<SKExecutionMode>,
    {
        self.state.execution_mode(mode);
        self
    }

    fn build(self) -> Result<SKSGDRegressor<F>, SKError> {
        sk_validate_sgd_configuration(&self.config)?;
        sk_validate_hyperparameter(
            "epsilon",
            self.epsilon >= F::zero() && self.epsilon.is_finite(),
            "must be non-negative and finite",
        )?;
        Ok(SKSGDRegressor {
            plan: self.state,
            config: self.config,
            loss: self.loss,
            epsilon: self.epsilon,
            warm_start: self.warm_start,
            backend: self.backend,
        })
    }
}
