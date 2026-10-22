//! The `SKLogisticRegression` estimator and its supervised fit (spec
//! `logistic-regression`).

use sciencekit_common::{
    SKBackendKind, SKBuilderState, SKDataView, SKError, SKExecutionContext, SKExecutionMode,
    SKExecutionPlan, SKFloat, SKOperationAttributes, SKOperationKind, SKSupervisedFit,
    SKSupervisedLazySource, SKTargetView, sk_run_operation,
};

use crate::logistic::SKLogisticState;
use crate::logistic::config::{
    SKClassWeight, SKLogisticConfig, SKLogisticPenalty, SKLogisticSolver, SKMultiClass,
};
use crate::logistic::resolution::sk_resolve_multi_class;
use crate::logistic::sk_canonicalize_targets;
use crate::logistic_regression::builder::SKLogisticRegressionBuilder;
use crate::logistic_regression::model::SKLogisticRegressionModel;

/// Logistic regression, builder-configured.
///
/// Immutable configuration carrying scalar values (`alpha`, `l1_ratio`,
/// `tolerance`), so the estimator is generic over the model scalar `F`. The
/// learned state lives only on the distinct [`SKLogisticRegressionModel`].
#[derive(Debug, Clone)]
pub struct SKLogisticRegression<F> {
    pub(super) plan: SKBuilderState,
    pub(super) config: SKLogisticConfig<F>,
    pub(super) warm_start_solution: Option<crate::logistic::SKLogisticSolution<F>>,
    pub(super) backend: SKBackendKind,
}

impl<F: SKFloat> SKLogisticRegression<F> {
    /// Start building an estimator (the only construction surface).
    pub fn build() -> SKLogisticRegressionBuilder<F> {
        SKLogisticRegressionBuilder::new()
    }

    /// The configured execution intent.
    pub fn execution_intent(&self) -> SKExecutionMode {
        self.plan.execution_intent()
    }

    /// Resolve the execution intent against a context.
    pub fn resolve_plan(&self, context: &SKExecutionContext) -> Result<SKExecutionPlan, SKError> {
        self.plan.resolve_plan(context)
    }

    /// The backend kind recorded in operation spans and transferred to models.
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

    /// The multi-class strategy.
    pub fn multi_class(&self) -> SKMultiClass {
        self.config.multi_class
    }

    /// The regularization strength `alpha`.
    pub fn alpha(&self) -> F {
        self.config.alpha
    }

    /// The inverse regularization strength `C = 1/alpha`.
    pub fn c(&self) -> F {
        F::one() / self.config.alpha
    }

    /// The elastic-net mixing parameter.
    pub fn l1_ratio(&self) -> F {
        self.config.l1_ratio
    }

    /// The class-weighting policy.
    pub fn class_weight(&self) -> &SKClassWeight<F> {
        &self.config.class_weight
    }

    /// Whether the fit solves for an unpenalized intercept.
    pub fn include_intercept(&self) -> bool {
        self.config.include_intercept
    }

    /// The convergence tolerance.
    pub fn tolerance(&self) -> F {
        self.config.tolerance
    }

    /// The iteration budget.
    pub fn maximum_number_of_iterations(&self) -> usize {
        self.config.maximum_number_of_iterations
    }

    /// Fit on a sequential supervised source, retaining each batch's design and
    /// solving the whole accumulated design once at the end.
    pub fn fit_streaming<S>(&self, source: &mut S) -> Result<SKLogisticRegressionModel<F>, SKError>
    where
        S: SKSupervisedLazySource<F> + Send,
        S::Error: From<SKError> + Into<SKError>,
    {
        crate::logistic_regression::streaming::sk_fit_streaming(
            source,
            &self.config,
            self.backend,
            self.execution_intent(),
        )
    }
}

impl<F: SKFloat> SKSupervisedFit<F> for SKLogisticRegression<F> {
    type Model = SKLogisticRegressionModel<F>;
    type Error = SKError;

    fn fit<'a, X, T>(&self, features: X, targets: T) -> Result<Self::Model, Self::Error>
    where
        X: TryInto<SKDataView<'a, F>, Error = SKError>,
        T: TryInto<SKTargetView<'a, F>, Error = SKError>,
    {
        let data = features.try_into()?;
        let target = targets.try_into()?;
        let (labels, classes) = sk_canonicalize_targets(target)?;
        let (rows_count, columns_count) = sk_dimensions(&data);
        sk_validate_shape(rows_count, columns_count, labels.len())?;
        let number_of_classes = classes.number_of_classes();
        let strategy = sk_resolve_multi_class(
            self.config.multi_class,
            self.config.solver,
            number_of_classes,
        )?;
        let execution_intent = self.execution_intent();
        let attributes = SKOperationAttributes {
            operation: SKOperationKind::Fit,
            rows_count,
            columns_count,
            execution_mode: execution_intent,
            backend: self.backend,
        };
        sk_run_operation(attributes, || {
            let mut state = match data {
                SKDataView::Dense(_) => SKLogisticState::new_dense(),
                SKDataView::Sparse(_) => SKLogisticState::new_sparse(),
                _ => return Err(unsupported_representation()),
            };
            match data {
                SKDataView::Dense(dense) => state.accumulate_dense(dense, &labels),
                SKDataView::Sparse(sparse) => state.accumulate_sparse(sparse, &labels),
                _ => return Err(unsupported_representation()),
            }
            let initial = if self.config.warm_start {
                self.warm_start_solution.as_ref()
            } else {
                None
            };
            let solution = state.solve(&self.config, strategy, number_of_classes, initial)?;
            Ok(SKLogisticRegressionModel::from_parts(
                solution,
                classes,
                strategy,
                number_of_classes,
                self.config.clone(),
                self.backend,
                execution_intent,
                state,
            ))
        })
    }
}

/// The row and column counts of a feature view.
pub(crate) fn sk_dimensions<F>(data: &SKDataView<'_, F>) -> (usize, usize) {
    match data {
        SKDataView::Dense(dense) => dense.dim(),
        SKDataView::Sparse(sparse) => (sparse.rows(), sparse.cols()),
        _ => (0, 0),
    }
}

/// Reject empty inputs and a label count that disagrees with the row count.
pub(crate) fn sk_validate_shape(
    rows_count: usize,
    columns_count: usize,
    labels_count: usize,
) -> Result<(), SKError> {
    if rows_count == 0 || columns_count == 0 {
        return Err(SKError::ShapeMismatch {
            expected: vec![rows_count.max(1), columns_count.max(1)],
            found: vec![rows_count, columns_count],
        });
    }
    if labels_count != rows_count {
        return Err(SKError::ShapeMismatch {
            expected: vec![rows_count],
            found: vec![labels_count],
        });
    }
    Ok(())
}

/// The error for a feature representation the estimators do not consume.
pub(crate) fn unsupported_representation() -> SKError {
    SKError::UnsupportedRepresentation {
        representation: "unknown",
        suggestion: "convert the features to a dense matrix or a CSR matrix",
    }
}
