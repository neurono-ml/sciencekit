//! The `SKSGDClassifier` estimator and its supervised fit (spec
//! `stochastic-gradient-classifier`).

use sciencekit_common::{
    SKBackendKind, SKBuilderState, SKDataView, SKError, SKExecutionContext, SKExecutionMode,
    SKExecutionPlan, SKFloat, SKOperationAttributes, SKOperationKind, SKSupervisedFit,
    SKSupervisedLazySource, SKTargetView, sk_run_operation,
};

use super::builder::SKSGDClassifierBuilder;
use super::coefficients::SKSGDClassifierCoefficients;
use super::fitting::{SKFeatureDesign, sk_create_states, sk_fit_epochs, sk_fit_streaming};
use super::model::SKSGDClassifierModel;
use super::targets::{sk_canonicalize_classifier_targets, sk_validate_number_of_classes};
use crate::class_weight::SKClassWeight;
use crate::sgd::SKSGDClassifierLoss;
use crate::sgd::config::SKSGDConfig;

/// Stochastic-gradient linear classifier, builder-configured.
///
/// Immutable configuration carrying scalar values, so the estimator is generic
/// over the model scalar `F`. The learned state lives only on the distinct
/// [`SKSGDClassifierModel`].
#[derive(Debug, Clone)]
pub struct SKSGDClassifier<F> {
    pub(super) plan: SKBuilderState,
    pub(super) config: SKSGDConfig<F>,
    pub(super) loss: SKSGDClassifierLoss,
    pub(super) class_weight: SKClassWeight<F>,
    pub(super) warm_start: Option<SKSGDClassifierCoefficients<F>>,
    pub(super) backend: SKBackendKind,
}

impl<F: SKFloat> SKSGDClassifier<F> {
    /// Start building an estimator (the only construction surface).
    pub fn build() -> SKSGDClassifierBuilder<F> {
        SKSGDClassifierBuilder::new()
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
    /// Every class must appear in the first non-empty batch. The model reports
    /// `epochs_taken() == 1` and `converged() == false`.
    pub fn fit_streaming<S>(&self, source: &mut S) -> Result<SKSGDClassifierModel<F>, SKError>
    where
        S: SKSupervisedLazySource<F> + Send,
        S::Error: From<SKError> + Into<SKError>,
    {
        sk_fit_streaming(
            source,
            &self.config,
            self.loss,
            &self.class_weight,
            self.warm_start.as_ref(),
            self.backend,
            self.execution_intent(),
        )
    }
}

impl<F: SKFloat> SKSupervisedFit<F> for SKSGDClassifier<F> {
    type Model = SKSGDClassifierModel<F>;
    type Error = SKError;

    fn fit<'a, X, T>(&self, features: X, targets: T) -> Result<Self::Model, Self::Error>
    where
        X: TryInto<SKDataView<'a, F>, Error = SKError>,
        T: TryInto<SKTargetView<'a, F>, Error = SKError>,
    {
        let data = features.try_into()?;
        let target = targets.try_into()?;
        let (class_indices, classes) = sk_canonicalize_classifier_targets(target)?;
        sk_validate_number_of_classes(&classes)?;
        let (rows_count, columns_count) = dimensions(&data);
        sk_validate_shape(rows_count, columns_count, class_indices.len())?;
        let execution_intent = self.execution_intent();
        let attributes = SKOperationAttributes {
            operation: SKOperationKind::Fit,
            rows_count,
            columns_count,
            execution_mode: execution_intent,
            backend: self.backend,
        };
        sk_run_operation(attributes, || {
            let design = SKFeatureDesign::from_view(data)?;
            let states = sk_create_states(
                &self.config,
                self.loss,
                self.warm_start.as_ref(),
                &classes,
                columns_count,
                &class_indices,
            )?;
            let outcome = sk_fit_epochs(
                &self.config,
                self.loss,
                &self.class_weight,
                states,
                &class_indices,
                classes.number_of_classes(),
                design,
            )?;
            Ok(SKSGDClassifierModel::from_parts(
                outcome.coefficients,
                outcome.intercepts,
                classes,
                outcome.states,
                &self.config,
                self.loss,
                &self.class_weight,
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

/// Reject empty inputs and a label count that disagrees with the row count.
fn sk_validate_shape(
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
