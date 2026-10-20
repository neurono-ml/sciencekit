//! The fitted `SKOneHotEncoderModel` and its dense transform.

use ndarray::Array2;
use sciencekit_common::{
    SKBackendKind, SKCategoricalView, SKError, SKExecutionMode, SKFloat, SKOperationAttributes,
    SKOperationKind, sk_run_operation,
};

use crate::one_hot_encoder::categories::SKOneHotColumnCategories;
use crate::one_hot_encoder::output::SKEncodedOutput;
use crate::one_hot_encoder::unknown_policy::SKUnknownPolicy;

/// Fitted one-hot encoder: the learned per-column category tables, the output
/// column offsets and the carried configuration. The tables are the export
/// boundary (they feed the future model header).
#[derive(Debug, Clone)]
pub struct SKOneHotEncoderModel {
    pub(super) columns: Vec<SKOneHotColumnCategories>,
    pub(super) offsets: Vec<usize>,
    pub(super) number_of_output_features: usize,
    pub(super) sparse_output: bool,
    pub(super) handle_unknown: SKUnknownPolicy,
    pub(super) backend: SKBackendKind,
    pub(super) execution_intent: SKExecutionMode,
}

impl SKOneHotEncoderModel {
    /// Wrap learned columns into a fitted model. Crate-internal: models come
    /// from `fit`/`fit_streaming`.
    pub(crate) fn from_columns(
        columns: Vec<SKOneHotColumnCategories>,
        sparse_output: bool,
        handle_unknown: SKUnknownPolicy,
        backend: SKBackendKind,
        execution_intent: SKExecutionMode,
    ) -> Self {
        let mut offsets = Vec::with_capacity(columns.len());
        let mut running = 0;
        for column in &columns {
            offsets.push(running);
            running += column.number_of_categories();
        }
        Self {
            number_of_output_features: running,
            columns,
            offsets,
            sparse_output,
            handle_unknown,
            backend,
            execution_intent,
        }
    }

    /// The number of input columns.
    pub fn number_of_input_columns(&self) -> usize {
        self.columns.len()
    }

    /// The total number of output (indicator) features.
    pub fn number_of_output_features(&self) -> usize {
        self.number_of_output_features
    }

    /// The category table of an input column.
    pub fn column_categories(&self, column: usize) -> Option<&SKOneHotColumnCategories> {
        self.columns.get(column)
    }

    /// The backend kind carried from the estimator that produced this model.
    pub fn backend(&self) -> SKBackendKind {
        self.backend
    }

    /// The unknown-category policy carried from the estimator.
    pub fn handle_unknown(&self) -> SKUnknownPolicy {
        self.handle_unknown
    }

    /// Whether the default transform returns a sparse matrix.
    pub fn is_sparse_output(&self) -> bool {
        self.sparse_output
    }

    /// Encode into the representation selected by the builder.
    pub fn transform<'a, F, C>(&self, categories: C) -> Result<SKEncodedOutput<F>, SKError>
    where
        F: SKFloat,
        C: TryInto<SKCategoricalView<'a>, Error = SKError>,
    {
        if self.sparse_output {
            Ok(SKEncodedOutput::Sparse(
                self.transform_sparse::<F, C>(categories)?,
            ))
        } else {
            Ok(SKEncodedOutput::Dense(
                self.transform_dense::<F, C>(categories)?,
            ))
        }
    }

    /// Encode into a dense indicator matrix.
    pub fn transform_dense<'a, F, C>(&self, categories: C) -> Result<Array2<F>, SKError>
    where
        F: SKFloat,
        C: TryInto<SKCategoricalView<'a>, Error = SKError>,
    {
        let view = categories.try_into()?;
        self.validate_shape(&view)?;
        sk_run_operation(self.operation_attributes(&view), || {
            let rows_count = view.number_of_rows();
            let mut output = Array2::<F>::zeros((rows_count, self.number_of_output_features));
            for row in 0..rows_count {
                for global_index in self.resolve_row(&view, row)? {
                    output[[row, global_index]] = F::one();
                }
            }
            Ok(output)
        })
    }

    /// The observability attributes for a transform over `view`.
    pub(super) fn operation_attributes(
        &self,
        view: &SKCategoricalView<'_>,
    ) -> SKOperationAttributes {
        SKOperationAttributes {
            operation: SKOperationKind::Transform,
            rows_count: view.number_of_rows(),
            columns_count: view.number_of_columns(),
            execution_mode: self.execution_intent,
            backend: self.backend,
        }
    }

    /// Reject a view whose column count differs from the fitted model, or a
    /// nominal matrix with a ragged row (shorter than its declared width).
    pub(super) fn validate_shape(&self, view: &SKCategoricalView<'_>) -> Result<(), SKError> {
        let found = view.number_of_columns();
        if found != self.columns.len() {
            return Err(SKError::shape_mismatch_2d(
                view.number_of_rows(),
                self.columns.len(),
                view.number_of_rows(),
                found,
            ));
        }
        if let SKCategoricalView::Nominal(matrix) = view {
            for row in 0..matrix.number_of_rows() {
                if matrix.row(row).len() != found {
                    return Err(SKError::ShapeMismatch {
                        expected: vec![matrix.number_of_rows(), found],
                        found: vec![matrix.number_of_rows(), matrix.row(row).len()],
                    });
                }
            }
        }
        Ok(())
    }
}
