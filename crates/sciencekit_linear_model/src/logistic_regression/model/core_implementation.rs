//! The fitted logistic-regression model: accessors (spec
//! `logistic-regression`).

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use sciencekit_common::{SKBackendKind, SKError, SKFloat, SKLabelTable};

use crate::logistic::SKLogisticState;
use crate::logistic::config::{
    SKClassWeight, SKLogisticConfig, SKLogisticPenalty, SKLogisticSolver,
};
use crate::logistic::resolution::SKResolvedMultiClass;
use crate::logistic::solver::SKLogisticSolution;

/// Fitted logistic-regression model: one coefficient vector per head, one
/// intercept per head, the class table and the retained design that backs
/// incremental updates.
///
/// Distinct type from the estimator ([`crate::logistic_regression::SKLogisticRegression`]):
/// the estimator is immutable configuration, the model is the learned state and
/// the export boundary. The model is `Send + Sync` by construction.
#[derive(Debug, Clone)]
pub struct SKLogisticRegressionModel<F> {
    pub(crate) coefficients: Array2<F>,
    pub(crate) intercepts: Array1<F>,
    pub(crate) classes: SKLabelTable,
    pub(crate) number_of_classes: usize,
    pub(crate) strategy: SKResolvedMultiClass,
    pub(crate) config: SKLogisticConfig<F>,
    pub(crate) backend: SKBackendKind,
    pub(crate) execution_intent: sciencekit_common::SKExecutionMode,
    pub(crate) state: SKLogisticState<F>,
}

impl<F: SKFloat> SKLogisticRegressionModel<F> {
    /// Assemble the model from a solve and the retained design.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn from_parts(
        solution: SKLogisticSolution<F>,
        classes: SKLabelTable,
        strategy: SKResolvedMultiClass,
        number_of_classes: usize,
        config: SKLogisticConfig<F>,
        backend: SKBackendKind,
        execution_intent: sciencekit_common::SKExecutionMode,
        state: SKLogisticState<F>,
    ) -> Self {
        Self {
            coefficients: solution.coefficients,
            intercepts: solution.intercepts,
            classes,
            number_of_classes,
            strategy,
            config,
            backend,
            execution_intent,
            state,
        }
    }

    /// The reversible class table.
    pub fn classes(&self) -> &SKLabelTable {
        &self.classes
    }

    /// The number of classes.
    pub fn number_of_classes(&self) -> usize {
        self.number_of_classes
    }

    /// The learned coefficients, shaped `heads × features`.
    pub fn coefficients(&self) -> ArrayView2<F> {
        self.coefficients.view()
    }

    /// The learned intercepts, one per head.
    pub fn intercepts(&self) -> ArrayView1<F> {
        self.intercepts.view()
    }

    /// The backend kind carried from the estimator that produced this model.
    pub fn backend(&self) -> SKBackendKind {
        self.backend
    }

    /// The regularization penalty.
    pub fn penalty(&self) -> SKLogisticPenalty {
        self.config.penalty
    }

    /// The optimization solver.
    pub fn solver(&self) -> SKLogisticSolver {
        self.config.solver
    }

    /// The elastic-net mixing parameter.
    pub fn l1_ratio(&self) -> F {
        self.config.l1_ratio
    }

    /// The regularization strength `alpha`.
    pub fn alpha(&self) -> F {
        self.config.alpha
    }

    /// The class-weighting policy.
    pub fn class_weight(&self) -> &SKClassWeight<F> {
        &self.config.class_weight
    }

    /// Whether the model reports an intercept.
    pub fn include_intercept(&self) -> bool {
        self.config.include_intercept
    }

    /// The number of features the model was fitted on.
    pub fn feature_count(&self) -> usize {
        self.coefficients.ncols()
    }

    /// The number of rows the model has learned from.
    pub fn row_count(&self) -> u64 {
        self.state.row_count()
    }

    /// Whether a binary fit produced a single head.
    #[allow(dead_code)]
    pub(crate) fn is_binary(&self) -> bool {
        self.strategy == SKResolvedMultiClass::Binary
    }

    /// The error for a prediction whose column count disagrees with the fit.
    pub(crate) fn feature_mismatch(&self, rows: usize, columns: usize) -> SKError {
        SKError::shape_mismatch_2d(rows, self.feature_count(), rows, columns)
    }
}
