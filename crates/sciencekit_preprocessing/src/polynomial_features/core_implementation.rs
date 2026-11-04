//! The `SKPolynomialFeatures` estimator: builder entry, dense `fit` and
//! out-of-core `fit_streaming`.

use sciencekit_common::{
    SKBackendKind, SKBuilderState, SKDataBatch, SKDataView, SKError, SKExecutionContext,
    SKExecutionMode, SKExecutionPlan, SKFloat, SKOperationAttributes, SKOperationKind,
    SKSourceProvider,
    execution::{
        SKAccessPattern, SKStreamDecision, SKStreamingPassPolicy, sk_run_streaming_provider,
    },
    sk_run_operation,
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
        let (rows_count, columns_count) = match features.try_into()? {
            SKDataView::Dense(dense) => dense.dim(),
            SKDataView::Sparse(sparse) => (sparse.rows(), sparse.cols()),
            _ => {
                return Err(SKError::UnsupportedRepresentation {
                    representation: "unknown",
                    suggestion: "convert the features to a dense matrix or a CSR matrix",
                });
            }
        };
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
    pub fn fit_streaming<F, Provider>(
        &self,
        mut provider: Provider,
    ) -> Result<SKPolynomialFeaturesModel, SKError>
    where
        F: SKFloat,
        Provider: SKSourceProvider<F>,
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
            let mut state = SKPolynomialFeaturesStreamState::new();
            sk_run_streaming_provider(
                &mut provider,
                &mut state,
                |batch, parallelism, state| state.update(batch, parallelism),
                &plan,
                SKStreamingPassPolicy::SinglePass,
            )?;
            if let Some(error) = state.error.take() {
                return Err(error);
            }
            let Some(columns_count) = state.columns_count else {
                return Err(SKError::Conversion(
                    "streaming fit received no batches".into(),
                ));
            };
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

/// The expander's single-pass accumulator: the observed column count. The
/// expansion depends only on that count, whichever representation carries it.
struct SKPolynomialFeaturesStreamState {
    columns_count: Option<usize>,
    error: Option<SKError>,
}

impl SKPolynomialFeaturesStreamState {
    fn new() -> Self {
        Self {
            columns_count: None,
            error: None,
        }
    }

    fn update<F: SKFloat>(
        &mut self,
        batch: SKDataBatch<F>,
        _parallelism: usize,
    ) -> SKStreamDecision {
        match self.accumulate(batch) {
            Ok(()) => SKStreamDecision::Continue,
            Err(error) => {
                self.error = Some(error);
                SKStreamDecision::Stop
            }
        }
    }

    fn accumulate<F: SKFloat>(&mut self, batch: SKDataBatch<F>) -> Result<(), SKError> {
        let rows_count = batch.number_of_rows();
        let columns_count = batch.number_of_columns();
        if rows_count == 0 {
            return Ok(());
        }
        if columns_count == 0 {
            return Err(SKError::shape_mismatch_2d(rows_count, 1, rows_count, 0));
        }
        match self.columns_count {
            None => self.columns_count = Some(columns_count),
            Some(expected) if expected != columns_count => {
                return Err(SKError::ShapeMismatch {
                    expected: vec![expected],
                    found: vec![columns_count],
                });
            }
            _ => {}
        }
        Ok(())
    }
}
