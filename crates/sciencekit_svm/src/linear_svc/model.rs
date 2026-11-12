//! Fitted [`SKLinearSVCModel`] and its per-head coefficient state
//! (spec `linear-support-vector-classifier`).

use ndarray::Array1;
use sciencekit_common::{SKBackendKind, SKBuilderState, SKExecutionMode, SKFloat, SKLabelTable};

use super::config::SKLinearSvcLoss;

/// One fitted binary head: a coefficient vector and an intercept.
#[derive(Debug, Clone)]
pub struct SKLinearSvcHead<F: SKFloat> {
    pub(crate) coefficients: Array1<F>,
    pub(crate) intercept: F,
    pub(crate) iterations: usize,
}

impl<F: SKFloat> SKLinearSvcHead<F> {
    /// Coefficient vector in feature order (one entry per input column).
    pub fn coefficients(&self) -> &Array1<F> {
        &self.coefficients
    }

    /// Recovered intercept (`0` when the model was fitted without one).
    pub fn intercept(&self) -> F {
        self.intercept
    }

    /// Dual sweeps executed for this head.
    pub fn iterations(&self) -> usize {
        self.iterations
    }
}

/// Fitted linear support-vector classifier model.
///
/// A binary model retains one head whose positive side is the larger canonical
/// class index; a multiclass model retains one head per class in canonical
/// order.
#[derive(Debug, Clone)]
pub struct SKLinearSVCModel<F: SKFloat> {
    pub(crate) classes: SKLabelTable,
    pub(crate) heads: Vec<SKLinearSvcHead<F>>,
    pub(crate) loss: SKLinearSvcLoss,
    pub(crate) regularization_c: F,
    pub(crate) number_of_features: usize,
    pub(crate) include_intercept: bool,
    pub(crate) backend: SKBackendKind,
    pub(crate) builder_state: SKBuilderState,
}

impl<F: SKFloat> SKLinearSVCModel<F> {
    /// Canonical class label table.
    pub fn classes(&self) -> &SKLabelTable {
        &self.classes
    }

    /// Number of classes in the fitted problem.
    pub fn number_of_classes(&self) -> usize {
        self.classes.number_of_classes()
    }

    /// Number of input features seen at fit time.
    pub fn number_of_features(&self) -> usize {
        self.number_of_features
    }

    /// Fitted heads in canonical class order (one head for a binary problem).
    pub fn heads(&self) -> &[SKLinearSvcHead<F>] {
        &self.heads
    }

    /// Loss function the model was fitted with.
    pub fn loss(&self) -> SKLinearSvcLoss {
        self.loss
    }

    /// Regularization strength `C` the model was fitted with.
    #[allow(non_snake_case)]
    pub fn C(&self) -> F {
        self.regularization_c
    }

    /// Whether the model carries an intercept.
    pub fn include_intercept(&self) -> bool {
        self.include_intercept
    }

    /// Maximum number of dual sweeps executed across the fitted heads.
    pub fn iterations(&self) -> usize {
        self.heads
            .iter()
            .map(SKLinearSvcHead::iterations)
            .max()
            .unwrap_or(0)
    }

    /// Whether every head reached its tolerance. `fit` returns
    /// `NotConverged` instead of building a model otherwise.
    pub fn converged(&self) -> bool {
        true
    }

    /// Configured math backend kind transferred from the estimator.
    pub fn backend(&self) -> SKBackendKind {
        self.backend
    }

    /// Configured execution mode transferred from the estimator.
    pub fn execution_mode(&self) -> SKExecutionMode {
        self.builder_state.execution_intent()
    }
}
