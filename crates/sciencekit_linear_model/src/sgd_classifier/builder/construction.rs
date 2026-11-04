//! Construction and validation of the SGD classifier builder (spec
//! `stochastic-gradient-classifier`).

use sciencekit_common::{
    SKBackendKind, SKBuilderState, SKError, SKExecutionMode, SKFloat,
    builders::{SKBuilder, sk_validate_hyperparameter},
};

use super::super::coefficients::SKSGDClassifierCoefficients;
use super::super::core_implementation::SKSGDClassifier;
use crate::class_weight::SKClassWeight;
use crate::sgd::config::{SKSGDConfig, sk_validate_sgd_configuration};
use crate::sgd::{SKSGDClassifierLoss, SKSGDLearningRate};

/// Builder for [`SKSGDClassifier`]: the only construction surface.
pub struct SKSGDClassifierBuilder<F> {
    pub(super) state: SKBuilderState,
    pub(super) config: SKSGDConfig<F>,
    pub(super) loss: SKSGDClassifierLoss,
    pub(super) class_weight: SKClassWeight<F>,
    pub(super) warm_start: Option<SKSGDClassifierCoefficients<F>>,
    pub(super) backend: SKBackendKind,
}

impl<F: SKFloat> SKSGDClassifierBuilder<F> {
    /// A fresh builder with the documented defaults.
    pub(crate) fn new() -> SKSGDClassifierBuilder<F> {
        Self {
            state: SKBuilderState::new(),
            config: SKSGDConfig::<F> {
                learning_rate: SKSGDLearningRate::Optimal,
                ..SKSGDConfig::default()
            },
            loss: SKSGDClassifierLoss::Hinge,
            class_weight: SKClassWeight::None,
            warm_start: None,
            backend: SKBackendKind::Faer,
        }
    }
}

impl<F: SKFloat> Default for SKSGDClassifierBuilder<F> {
    fn default() -> Self {
        Self::new()
    }
}

impl<F: SKFloat> SKBuilder<SKSGDClassifier<F>> for SKSGDClassifierBuilder<F> {
    fn execution_mode<E>(&mut self, mode: E) -> &mut Self
    where
        E: TryInto<SKExecutionMode>,
    {
        self.state.execution_mode(mode);
        self
    }

    fn build(self) -> Result<SKSGDClassifier<F>, SKError> {
        sk_validate_sgd_configuration(&self.config)?;
        sk_validate_hyperparameter(
            "class_weight",
            !matches!(self.class_weight, SKClassWeight::Explicit(ref weights) if weights.is_empty()),
            "explicit class weights must not be empty",
        )?;
        Ok(SKSGDClassifier {
            plan: self.state,
            config: self.config,
            loss: self.loss,
            class_weight: self.class_weight,
            warm_start: self.warm_start,
            backend: self.backend,
        })
    }
}
