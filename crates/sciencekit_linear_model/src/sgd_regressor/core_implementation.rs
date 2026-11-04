//! The `SKSGDRegressor` estimator and its supervised fit (spec
//! `stochastic-gradient-regressor`).

use sciencekit_common::{
    SKBackendKind, SKBuilderState, SKDataView, SKError, SKExecutionContext, SKExecutionMode,
    SKExecutionPlan, SKFloat, SKOperationAttributes, SKOperationKind, SKSupervisedFit,
    SKSupervisedLazySource, SKTargetView, sk_run_operation,
};

use super::builder::SKSGDRegressorBuilder;
use super::coefficients::SKSGDRegressorCoefficients;
use super::fitting::{SKRegressorDesign, sk_create_state, sk_fit_epochs, sk_fit_streaming};
use super::model::SKSGDRegressorModel;
use super::targets::sk_loss_adapter;
use crate::sgd::SKSGDRegressorLoss;
use crate::sgd::config::SKSGDConfig;

/// Stochastic-gradient linear regressor, builder-configured.
///
/// Immutable configuration carrying scalar values, so the estimator is generic
/// over the model scalar `F`. The learned state lives only on the distinct
/// [`SKSGDRegressorModel`].
#[derive(Debug, Clone)]
pub struct SKSGDRegressor<F> {
    pub(super) plan: SKBuilderState,
    pub(super) config: SKSGDConfig<F>,
    pub(super) loss: SKSGDRegressorLoss,
    pub(super) epsilon: F,
    pub(super) warm_start: Option<SKSGDRegressorCoefficients<F>>,
    pub(super) backend: SKBackendKind,
}

impl<F: SKFloat> SKSGDRegressor<F> {
    /// Start building an estimator (the only construction surface).
    pub fn build() -> SKSGDRegressorBuilder<F> {
        SKSGDRegressorBuilder::new()
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

    /// Fit on a one-shot supervised source in a single pass, retaining only one
    /// batch plus the coefficient state.
    ///
    /// The model reports `epochs_taken() == 1` and `converged() == false`.
    pub fn fit_streaming<S>(&self, source: &mut S) -> Result<SKSGDRegressorModel<F>, SKError>
    where
        S: SKSupervisedLazySource<F> + Send,
        S::Error: From<SKError> + Into<SKError>,
    {
        sk_fit_streaming(
            source,
            &self.config,
            self.loss,
            self.epsilon,
            self.warm_start.as_ref(),
            self.backend,
            self.execution_intent(),
        )
    }
}

impl<F: SKFloat> SKSupervisedFit<F> for SKSGDRegressor<F> {
    type Model = SKSGDRegressorModel<F>;
    type Error = SKError;

    fn fit<'a, X, T>(&self, features: X, targets: T) -> Result<Self::Model, Self::Error>
    where
        X: TryInto<SKDataView<'a, F>, Error = SKError>,
        T: TryInto<SKTargetView<'a, F>, Error = SKError>,
    {
        let data = features.try_into()?;
        let target = targets.try_into()?.as_continuous()?.into_owned();
        let (rows_count, columns_count) = dimensions(&data);
        sk_validate_shape(rows_count, columns_count, target.len())?;
        let execution_intent = self.execution_intent();
        let attributes = SKOperationAttributes {
            operation: SKOperationKind::Fit,
            rows_count,
            columns_count,
            execution_mode: execution_intent,
            backend: self.backend,
        };
        sk_run_operation(attributes, || {
            let design = SKRegressorDesign::from_view(data)?;
            let loss = sk_loss_adapter(self.loss, self.epsilon);
            let state = sk_create_state(
                &self.config,
                self.loss,
                self.epsilon,
                self.warm_start.as_ref(),
                columns_count,
                target
                    .as_slice()
                    .expect("a continuous target is contiguous"),
            )?;
            let outcome = sk_fit_epochs(
                &self.config,
                &loss,
                state,
                design,
                target.as_slice().expect("contiguous"),
            )?;
            Ok(SKSGDRegressorModel::from_parts(
                outcome.coefficients,
                outcome.intercept,
                outcome.state,
                &self.config,
                self.loss,
                self.epsilon,
                self.backend,
                execution_intent,
                outcome.converged,
                outcome.epochs_taken,
            ))
        })
    }
}

/// The row and column counts of a feature view.
fn dimensions<F>(data: &SKDataView<'_, F>) -> (usize, usize) {
    match data {
        SKDataView::Dense(dense) => dense.dim(),
        SKDataView::Sparse(sparse) => (sparse.rows(), sparse.cols()),
        _ => (0, 0),
    }
}

/// Reject empty inputs and a target count that disagrees with the row count.
fn sk_validate_shape(
    rows_count: usize,
    columns_count: usize,
    target_count: usize,
) -> Result<(), SKError> {
    if rows_count == 0 || columns_count == 0 {
        return Err(SKError::ShapeMismatch {
            expected: vec![rows_count.max(1), columns_count.max(1)],
            found: vec![rows_count, columns_count],
        });
    }
    if target_count != rows_count {
        return Err(SKError::ShapeMismatch {
            expected: vec![rows_count],
            found: vec![target_count],
        });
    }
    Ok(())
}
