//! The `SKOneHotEncoder` estimator: builder entry, dense `fit` and out-of-core
//! `fit_streaming`.

use sciencekit_common::{
    SKBackendKind, SKBuilderState, SKCategoricalView, SKError, SKExecutionMode,
    SKOperationAttributes, SKOperationKind, sk_run_operation,
};

use crate::one_hot_encoder::builder::SKOneHotEncoderBuilder;
use crate::one_hot_encoder::categories::{SKOneHotColumnCategories, build_column_categories};
use crate::one_hot_encoder::model::SKOneHotEncoderModel;
use crate::one_hot_encoder::streaming::accumulate_batches;
use crate::one_hot_encoder::unknown_policy::SKUnknownPolicy;

/// The configured one-hot encoder. Immutable configuration; the fitted
/// [`SKOneHotEncoderModel`] is the learned state.
pub struct SKOneHotEncoder {
    pub(super) state: SKBuilderState,
    pub(super) sparse_output: bool,
    pub(super) handle_unknown: SKUnknownPolicy,
    pub(super) backend: SKBackendKind,
}

impl SKOneHotEncoder {
    /// Start building an encoder.
    pub fn build() -> SKOneHotEncoderBuilder {
        SKOneHotEncoderBuilder::new()
    }

    /// The configured execution intent.
    pub fn execution_intent(&self) -> SKExecutionMode {
        self.state.execution_intent()
    }

    /// The backend kind recorded in spans and carried into the model.
    pub fn backend(&self) -> SKBackendKind {
        self.backend
    }

    /// The configured unknown-category policy.
    pub fn handle_unknown(&self) -> SKUnknownPolicy {
        self.handle_unknown
    }

    /// Whether the default transform returns a sparse matrix.
    pub fn sparse_output(&self) -> bool {
        self.sparse_output
    }

    /// Fit the encoder on a categorical view, learning the per-column category
    /// tables.
    pub fn fit<'a, C>(&self, categories: C) -> Result<SKOneHotEncoderModel, SKError>
    where
        C: TryInto<SKCategoricalView<'a>, Error = SKError>,
    {
        let view = categories.try_into()?;
        let attributes = SKOperationAttributes {
            operation: SKOperationKind::Fit,
            rows_count: view.number_of_rows(),
            columns_count: view.number_of_columns(),
            execution_mode: self.execution_intent(),
            backend: self.backend,
        };
        sk_run_operation(attributes, || {
            let columns = build_column_categories(&view)?;
            Ok(self.model_from_columns(columns))
        })
    }

    /// Fit on a sequence of categorical batches, equivalent to the dense `fit`
    /// on the concatenated rows: category indices follow first-occurrence order
    /// across the concatenation.
    pub fn fit_streaming<'a, I, C>(&self, batches: I) -> Result<SKOneHotEncoderModel, SKError>
    where
        I: IntoIterator<Item = C>,
        C: TryInto<SKCategoricalView<'a>, Error = SKError>,
    {
        let attributes = SKOperationAttributes {
            operation: SKOperationKind::Fit,
            rows_count: 0,
            columns_count: 0,
            execution_mode: self.execution_intent(),
            backend: self.backend,
        };
        sk_run_operation(attributes, || {
            let columns = accumulate_batches(batches)?;
            Ok(self.model_from_columns(columns))
        })
    }

    /// Wrap learned columns into a fitted model carrying this estimator's
    /// configuration.
    fn model_from_columns(&self, columns: Vec<SKOneHotColumnCategories>) -> SKOneHotEncoderModel {
        SKOneHotEncoderModel::from_columns(
            columns,
            self.sparse_output,
            self.handle_unknown,
            self.backend,
            self.execution_intent(),
        )
    }
}
