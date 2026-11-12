//! Builder-only construction for [`SKLinearSVC`]
//! (spec `linear-support-vector-classifier`).

use sciencekit_common::builders::sk_validate_hyperparameter;
use sciencekit_common::{
    SKBackendKind, SKBuilder, SKBuilderState, SKError, SKExecutionMode, SKFloat,
};

use super::config::{SKLinearSVCConfiguration, SKLinearSvcLoss};
use super::core_implementation::SKLinearSVC;
use crate::svc::{SKClassWeight, sk_validate_static_class_weight};

/// Builder for [`SKLinearSVC`].
pub struct SKLinearSVCBuilder<F: SKFloat> {
    configuration: SKLinearSVCConfiguration<F>,
}

impl<F: SKFloat> SKLinearSVCBuilder<F> {
    /// Create a builder carrying the documented defaults: `C = 1.0`,
    /// squared hinge, no class weights, intercept on, tolerance `1e-4`,
    /// `1000` iterations, `Faer` and `Automatic`.
    pub fn new() -> Self {
        Self {
            configuration: SKLinearSVCConfiguration {
                regularization_c: F::one(),
                loss: SKLinearSvcLoss::SquaredHinge,
                class_weight: SKClassWeight::None,
                include_intercept: true,
                tolerance: F::from(1e-4).expect("1e-4 fits SKFloat"),
                maximum_number_of_iterations: 1_000,
                backend: SKBackendKind::Faer,
                builder_state: SKBuilderState::new(),
            },
        }
    }

    /// Set the regularization strength `C` (default `1.0`).
    #[allow(non_snake_case)]
    pub fn C(mut self, value: F) -> Self {
        self.configuration.regularization_c = value;
        self
    }

    /// Set the loss function (default `SquaredHinge`).
    pub fn loss(mut self, value: SKLinearSvcLoss) -> Self {
        self.configuration.loss = value;
        self
    }

    /// Set the class-weighting policy (default `None`).
    pub fn class_weight(mut self, value: SKClassWeight<F>) -> Self {
        self.configuration.class_weight = value;
        self
    }

    /// Enable or disable the intercept (default `true`).
    pub fn include_intercept(mut self, value: bool) -> Self {
        self.configuration.include_intercept = value;
        self
    }

    /// Set the convergence tolerance on the maximum projected gradient
    /// (default `1e-4`).
    pub fn tolerance(mut self, value: F) -> Self {
        self.configuration.tolerance = value;
        self
    }

    /// Set the iteration budget in dual sweeps (default `1000`).
    pub fn maximum_number_of_iterations(mut self, value: usize) -> Self {
        self.configuration.maximum_number_of_iterations = value;
        self
    }

    /// Set the math backend kind (default `Faer`).
    pub fn backend(mut self, value: SKBackendKind) -> Self {
        self.configuration.backend = value;
        self
    }

    /// Set the execution mode intent (default `Automatic`).
    pub fn execution_mode<E>(mut self, mode: E) -> Self
    where
        E: TryInto<SKExecutionMode>,
    {
        self.configuration.builder_state.execution_mode(mode);
        self
    }
}

impl<F: SKFloat> Default for SKLinearSVCBuilder<F> {
    fn default() -> Self {
        Self::new()
    }
}

impl<F: SKFloat> SKBuilder<SKLinearSVC<F>> for SKLinearSVCBuilder<F> {
    fn execution_mode<E>(&mut self, mode: E) -> &mut Self
    where
        E: TryInto<SKExecutionMode>,
    {
        self.configuration.builder_state.execution_mode(mode);
        self
    }

    fn build(self) -> Result<SKLinearSVC<F>, SKError> {
        let config = &self.configuration;
        sk_validate_hyperparameter(
            "C",
            config.regularization_c > F::zero() && config.regularization_c.is_finite(),
            "regularization strength C must be positive and finite",
        )?;
        sk_validate_hyperparameter(
            "tolerance",
            config.tolerance > F::zero() && config.tolerance.is_finite(),
            "tolerance must be positive and finite",
        )?;
        sk_validate_hyperparameter(
            "maximum_number_of_iterations",
            config.maximum_number_of_iterations > 0,
            "maximum_number_of_iterations must be at least 1",
        )?;
        sk_validate_static_class_weight(&config.class_weight)?;
        Ok(SKLinearSVC::from_configuration(self.configuration))
    }
}
