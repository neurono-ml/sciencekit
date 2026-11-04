//! The fitted stochastic-gradient classifier model (spec
//! `stochastic-gradient-classifier`).

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use sciencekit_common::{SKBackendKind, SKError, SKExecutionMode, SKFloat, SKLabelTable};

use crate::class_weight::SKClassWeight;
use crate::sgd::config::SKSGDConfig;
use crate::sgd::{SKLearningState, SKSGDClassifierLoss};

/// The fitted stochastic-gradient linear classifier.
///
/// Distinct type from the estimator: the estimator is immutable configuration,
/// the model is the learned state and the export boundary. It carries the class
/// table, one coefficient vector and intercept per one-vs-rest head, the
/// learning states that back `partial_fit`, and the backend kind transferred by
/// `fit`. It is `Send + Sync` by construction.
#[derive(Debug, Clone)]
pub struct SKSGDClassifierModel<F> {
    pub(crate) coefficients: Array2<F>,
    pub(crate) intercepts: Array1<F>,
    pub(crate) classes: SKLabelTable,
    pub(crate) states: Vec<SKLearningState<F>>,
    pub(crate) config: SKSGDConfig<F>,
    pub(crate) loss: SKSGDClassifierLoss,
    pub(crate) class_weight: SKClassWeight<F>,
    pub(crate) backend: SKBackendKind,
    pub(crate) execution_intent: SKExecutionMode,
    pub(crate) converged: bool,
    pub(crate) epochs_taken: usize,
}

impl<F: SKFloat> SKSGDClassifierModel<F> {
    /// Assemble a model from its learned parts.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn from_parts(
        coefficients: Array2<F>,
        intercepts: Array1<F>,
        classes: SKLabelTable,
        states: Vec<SKLearningState<F>>,
        config: &SKSGDConfig<F>,
        loss: SKSGDClassifierLoss,
        class_weight: &SKClassWeight<F>,
        backend: SKBackendKind,
        execution_intent: SKExecutionMode,
        converged: bool,
        epochs_taken: usize,
    ) -> Self {
        Self {
            coefficients,
            intercepts,
            classes,
            states,
            config: config.clone(),
            loss,
            class_weight: class_weight.clone(),
            backend,
            execution_intent,
            converged,
            epochs_taken,
        }
    }

    /// The reversible class table, in first-occurrence canonical order.
    pub fn classes(&self) -> &SKLabelTable {
        &self.classes
    }

    /// The number of classes.
    pub fn number_of_classes(&self) -> usize {
        self.classes.number_of_classes()
    }

    /// The number of one-vs-rest heads (one for a binary fit).
    pub fn number_of_heads(&self) -> usize {
        self.coefficients.nrows()
    }

    /// The learned coefficients, shaped `heads × features`.
    pub fn coefficients(&self) -> ArrayView2<'_, F> {
        self.coefficients.view()
    }

    /// The learned intercepts, one per head.
    pub fn intercepts(&self) -> ArrayView1<'_, F> {
        self.intercepts.view()
    }

    /// The number of features the model was fitted on.
    pub fn feature_count(&self) -> usize {
        self.coefficients.ncols()
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

    /// The error for a prediction whose column count disagrees with the fit.
    pub(crate) fn feature_mismatch(&self, rows: usize, columns: usize) -> SKError {
        SKError::shape_mismatch_2d(rows, self.feature_count(), rows, columns)
    }

    /// Whether a binary fit produced a single head.
    pub(crate) fn is_binary(&self) -> bool {
        self.coefficients.nrows() == 1
    }
}
