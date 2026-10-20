//! The `SKPolynomialFeatures` estimator: builder entry, dense `fit` and
//! out-of-core `fit_streaming`.

use ndarray::{Array2, ArrayView2, Axis};
use sciencekit_common::{
    SKBackendKind, SKBuilderState, SKDataView, SKError, SKExecutionContext, SKExecutionMode,
    SKExecutionPlan, SKFloat, SKLazySource, SKOperationAttributes, SKOperationKind,
    execution::SKAccessPattern, sk_run_operation,
};

use crate::polynomial_features::builder::SKPolynomialFeaturesBuilder;
use crate::polynomial_features::model::SKPolynomialFeaturesModel;

/// The configured polynomial-features expander. Immutable configuration; the
/// fitted [`SKPolynomialFeaturesModel`] is the learned state.
pub struct SKPolynomialFeatures {
    pub(super) state: SKBuilderState,
    pub(super) degree: usize,
    pub(super) interaction_only: bool,
    pub(super) include_bias: bool,
    pub(super) sparse_output: bool,
    pub(super) backend: SKBackendKind,
}

impl SKPolynomialFeatures {
    /// Start building an expander.
    pub fn build() -> SKPolynomialFeaturesBuilder {
        SKPolynomialFeaturesBuilder::new()
    }

    /// The configured execution intent.
    pub fn execution_intent(&self) -> SKExecutionMode {
        self.state.execution_intent()
    }

    /// The backend kind recorded in spans and carried into the model.
    pub fn backend(&self) -> SKBackendKind {
        self.backend
    }

    /// Fit the expander: learn the input feature count and the expansion shape.
    pub fn fit<'a, F, X>(&self, features: X) -> Result<SKPolynomialFeaturesModel, SKError>
    where
        F: SKFloat,
        X: TryInto<SKDataView<'a, F>, Error = SKError>,
    {
        let dense = features.try_into()?.as_dense()?;
        let (rows_count, columns_count) = dense.dim();
        if rows_count == 0 || columns_count == 0 {
            return Err(SKError::shape_mismatch_2d(
                rows_count.max(1),
                columns_count.max(1),
                rows_count,
                columns_count,
            ));
        }
        let attributes = SKOperationAttributes {
            operation: SKOperationKind::Fit,
            rows_count,
            columns_count,
            execution_mode: self.execution_intent(),
            backend: self.backend,
        };
        sk_run_operation(attributes, || {
            Ok(self.model_from_input_features(columns_count))
        })
    }

    /// Fit on a sequential batch source, learning the same expansion the dense
    /// `fit` would learn for the concatenated data.
    pub fn fit_streaming<F, S>(&self, source: &mut S) -> Result<SKPolynomialFeaturesModel, SKError>
    where
        F: SKFloat,
        S: SKLazySource<F> + Send,
        S::Error: From<SKError> + Into<SKError>,
    {
        let attributes = SKOperationAttributes {
            operation: SKOperationKind::Fit,
            rows_count: 0,
            columns_count: 0,
            execution_mode: self.execution_intent(),
            backend: self.backend,
        };
        sk_run_operation(attributes, || {
            let plan = self.resolve_streaming_plan()?;
            let in_process = plan.mode == SKExecutionMode::InProcessSynchronous
                || plan.mode == SKExecutionMode::InProcessAsynchronous;
            let mut columns_count: Option<usize> = None;
            let mut collected: Vec<Array2<F>> = Vec::new();
            while let Some(batch) = source.next_batch().map_err(Into::into)? {
                let data = batch.data();
                if data.nrows() == 0 {
                    continue;
                }
                if data.ncols() == 0 {
                    return Err(SKError::shape_mismatch_2d(data.nrows(), 1, data.nrows(), 0));
                }
                match columns_count {
                    None => columns_count = Some(data.ncols()),
                    Some(expected) if expected != data.ncols() => {
                        return Err(SKError::ShapeMismatch {
                            expected: vec![expected],
                            found: vec![data.ncols()],
                        });
                    }
                    _ => {}
                }
                if in_process {
                    collected.push(data.to_owned());
                }
            }
            let Some(columns_count) = columns_count else {
                return Err(SKError::Conversion(
                    "streaming fit received no batches".into(),
                ));
            };
            if in_process {
                let views: Vec<ArrayView2<F>> =
                    collected.iter().map(|batch| batch.view()).collect();
                let concatenated = ndarray::concatenate(Axis(0), &views)
                    .map_err(|_| SKError::Conversion("batch concatenation failed".into()))?;
                return Ok(self.model_from_input_features(concatenated.ncols()));
            }
            Ok(self.model_from_input_features(columns_count))
        })
    }

    /// Build a fitted model from the learned input feature count.
    fn model_from_input_features(
        &self,
        number_of_input_features: usize,
    ) -> SKPolynomialFeaturesModel {
        SKPolynomialFeaturesModel::from_configuration(
            number_of_input_features,
            self.degree,
            self.interaction_only,
            self.include_bias,
            self.sparse_output,
            self.backend,
            self.execution_intent(),
        )
    }

    /// Resolve the plan for an unbounded stream (Sequential access pattern).
    fn resolve_streaming_plan(&self) -> Result<SKExecutionPlan, SKError> {
        let mut context = SKExecutionContext::real();
        context.dataset_size_bytes = u64::MAX;
        context.scalar_size_bytes = 8;
        context.dataset_elements = None;
        context.access_pattern = SKAccessPattern::Sequential;
        self.state.resolve_plan(&context)
    }
}
