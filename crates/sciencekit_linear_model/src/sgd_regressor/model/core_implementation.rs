//! The fitted stochastic-gradient regressor model (spec
//! `stochastic-gradient-regressor`).

use ndarray::{Array1, ArrayView1};
use sciencekit_common::{SKBackendKind, SKError, SKExecutionMode, SKFloat};

use crate::sgd::config::SKSGDConfig;
use crate::sgd::{SKLearningState, SKSGDRegressorLoss};

/// The fitted stochastic-gradient linear regressor.
///
/// Distinct type from the estimator: the estimator is immutable configuration,
/// the model is the learned state and the export boundary. It carries one
/// coefficient vector, the intercept, the learning state that backs
/// `partial_fit`, and the backend kind transferred by `fit`. It is `Send +
/// Sync` by construction.
#[derive(Debug, Clone)]
pub struct SKSGDRegressorModel<F> {
    pub(crate) coefficients: Array1<F>,
    pub(crate) intercept: F,
    pub(crate) state: SKLearningState<F>,
    pub(crate) config: SKSGDConfig<F>,
    pub(crate) loss: SKSGDRegressorLoss,
    pub(crate) epsilon: F,
    pub(crate) backend: SKBackendKind,
    pub(crate) execution_intent: SKExecutionMode,
    pub(crate) converged: bool,
    pub(crate) epochs_taken: usize,
}

impl<F: SKFloat> SKSGDRegressorModel<F> {
    /// Assemble a model from its learned parts.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn from_parts(
        coefficients: Array1<F>,
        intercept: F,
        state: SKLearningState<F>,
        config: &SKSGDConfig<F>,
        loss: SKSGDRegressorLoss,
        epsilon: F,
        backend: SKBackendKind,
        execution_intent: SKExecutionMode,
        converged: bool,
        epochs_taken: usize,
    ) -> Self {
        Self {
            coefficients,
            intercept,
            state,
            config: config.clone(),
            loss,
            epsilon,
            backend,
            execution_intent,
            converged,
            epochs_taken,
        }
    }

    /// The learned coefficients, one per feature.
    pub fn coefficients(&self) -> ArrayView1<'_, F> {
        self.coefficients.view()
    }

    /// The learned intercept (zero when `include_intercept` is false).
    pub fn intercept(&self) -> F {
        self.intercept
    }

    /// The number of features the model was fitted on.
    pub fn feature_count(&self) -> usize {
        self.coefficients.len()
    }

    /// The backend kind carried from the estimator that produced this model.
    pub fn backend(&self) -> SKBackendKind {
        self.backend
    }

    /// Whether the fit met the convergence criterion.
    pub fn converged(&self) -> bool {
        self.converged
    }

    /// The number of epochs (or streaming passes) the fit executed.
    pub fn epochs_taken(&self) -> usize {
        self.epochs_taken
    }
}

impl<F: SKFloat> SKSGDRegressorModel<F> {
    /// The error for a prediction whose column count disagrees with the fit.
    pub(crate) fn feature_mismatch(&self, rows: usize, columns: usize) -> SKError {
        SKError::shape_mismatch_2d(rows, self.feature_count(), rows, columns)
    }
}
