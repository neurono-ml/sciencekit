//! Configured [`SKLinearSVC`] estimator and hyperparameter accessors
//! (spec `linear-support-vector-classifier`).

use sciencekit_common::{SKBackendKind, SKExecutionMode, SKFloat};

use super::builder::SKLinearSVCBuilder;
use super::config::{SKLinearSVCConfiguration, SKLinearSVCLoss};
use crate::svc::SKClassWeight;

/// L2-regularized linear support-vector classifier.
///
/// Fitted through a LIBLINEAR-style dual coordinate-descent solver on the
/// configured loss, with one-vs-rest multiclass heads. Construct exclusively
/// through [`SKLinearSVC::build`].
#[derive(Debug, Clone)]
pub struct SKLinearSVC<F: SKFloat> {
    pub(crate) configuration: SKLinearSVCConfiguration<F>,
}

impl<F: SKFloat> SKLinearSVC<F> {
    pub(crate) fn from_configuration(configuration: SKLinearSVCConfiguration<F>) -> Self {
        Self { configuration }
    }

    /// Start building an [`SKLinearSVC`] with documented defaults.
    pub fn build() -> SKLinearSVCBuilder<F> {
        SKLinearSVCBuilder::new()
    }

    /// Configured regularization strength `C`.
    #[allow(non_snake_case)]
    pub fn C(&self) -> F {
        self.configuration.regularization_c
    }

    /// Configured loss function.
    pub fn loss(&self) -> SKLinearSVCLoss {
        self.configuration.loss
    }

    /// Configured class-weighting policy.
    pub fn class_weight(&self) -> &SKClassWeight<F> {
        &self.configuration.class_weight
    }

    /// Whether an intercept is fitted.
    pub fn include_intercept(&self) -> bool {
        self.configuration.include_intercept
    }

    /// Configured convergence tolerance on the maximum projected gradient.
    pub fn tolerance(&self) -> F {
        self.configuration.tolerance
    }

    /// Configured number of dual coordinate-descent sweeps.
    pub fn maximum_number_of_iterations(&self) -> usize {
        self.configuration.maximum_number_of_iterations
    }

    /// Configured math backend kind.
    pub fn backend(&self) -> SKBackendKind {
        self.configuration.backend
    }

    /// Configured execution mode.
    pub fn execution_mode(&self) -> SKExecutionMode {
        self.configuration.builder_state.execution_intent()
    }
}
