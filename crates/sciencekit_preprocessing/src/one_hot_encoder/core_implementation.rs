//! The `SKOneHotEncoder` estimator: builder entry, dense `fit` and out-of-core
//! `fit_streaming`.

use sciencekit_common::{
    SKBackendKind, SKBuilderState, SKCategoricalView, SKError, SKExecutionMode,
    SKOperationAttributes, SKOperationKind, sk_run_operation,
};

use crate::one_hot_encoder::builder::SKOneHotEncoderBuilder;
use crate::one_hot_encoder::categories::{SKIntegerCategoryTable, build_column_categories};
use crate::one_hot_encoder::model::SKOneHotEncoderModel;
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
            let accumulator = accumulate_batches(batches)?;
            let columns = accumulator.finalize()?;
            Ok(self.model_from_columns(columns))
        })
    }

    /// Wrap learned columns into a fitted model carrying this estimator's
    /// configuration.
    fn model_from_columns(
        &self,
        columns: Vec<crate::one_hot_encoder::categories::SKOneHotColumnCategories>,
    ) -> SKOneHotEncoderModel {
        SKOneHotEncoderModel::from_columns(
            columns,
            self.sparse_output,
            self.handle_unknown,
            self.backend,
            self.execution_intent(),
        )
    }
}

/// Running accumulator for `fit_streaming`: one column of labels (nominal) or
/// one code table (integer) per input column, in batch order.
enum StreamingCategories {
    Nominal(Vec<Vec<String>>),
    Integer(Vec<SKIntegerCategoryTable>),
}

impl StreamingCategories {
    fn finalize(
        self,
    ) -> Result<Vec<crate::one_hot_encoder::categories::SKOneHotColumnCategories>, SKError> {
        use crate::one_hot_encoder::categories::SKOneHotColumnCategories;
        use sciencekit_common::sk_canonicalize_labels;
        match self {
            StreamingCategories::Nominal(columns) => Ok(columns
                .iter()
                .map(|labels| {
                    let borrowed: Vec<&str> = labels.iter().map(String::as_str).collect();
                    let (_indices, table) = sk_canonicalize_labels(&borrowed);
                    SKOneHotColumnCategories::Nominal(table)
                })
                .collect()),
            StreamingCategories::Integer(tables) => Ok(tables
                .into_iter()
                .map(SKOneHotColumnCategories::Integer)
                .collect()),
        }
    }
}

/// Accumulate every batch into running column state, enforcing a consistent
/// column count and representation.
fn accumulate_batches<'a, I, C>(batches: I) -> Result<StreamingCategories, SKError>
where
    I: IntoIterator<Item = C>,
    C: TryInto<SKCategoricalView<'a>, Error = SKError>,
{
    let mut accumulator: Option<StreamingCategories> = None;
    let mut number_of_columns: Option<usize> = None;

    for batch in batches {
        let view = batch.try_into()?;
        if view.is_empty() {
            continue;
        }
        let (rows_count, columns_count) = view.shape();
        match number_of_columns {
            None => number_of_columns = Some(columns_count),
            Some(expected) if expected != columns_count => {
                return Err(SKError::ShapeMismatch {
                    expected: vec![expected],
                    found: vec![columns_count],
                });
            }
            _ => {}
        }
        accumulate_one(&view, rows_count, columns_count, &mut accumulator)?;
    }

    accumulator.ok_or_else(|| SKError::Conversion("streaming fit received no batches".into()))
}

/// Fold one batch into the accumulator, initializing it from the first batch.
fn accumulate_one(
    view: &SKCategoricalView<'_>,
    rows_count: usize,
    columns_count: usize,
    accumulator: &mut Option<StreamingCategories>,
) -> Result<(), SKError> {
    match view {
        SKCategoricalView::Nominal(matrix) => {
            let target = accumulator.get_or_insert_with(|| {
                StreamingCategories::Nominal(vec![Vec::new(); columns_count])
            });
            let StreamingCategories::Nominal(columns) = target else {
                return Err(representation_changed("nominal"));
            };
            for (column, labels) in columns.iter_mut().enumerate() {
                for row in 0..rows_count {
                    labels.push(matrix.value(row, column).to_owned());
                }
            }
        }
        SKCategoricalView::Integer(codes) => {
            let target = accumulator.get_or_insert_with(|| {
                StreamingCategories::Integer(vec![SKIntegerCategoryTable::default(); columns_count])
            });
            let StreamingCategories::Integer(tables) = target else {
                return Err(representation_changed("integer"));
            };
            for (column, table) in tables.iter_mut().enumerate() {
                for row in 0..rows_count {
                    let code = codes[[row, column]];
                    if code != sciencekit_common::SK_CATEGORICAL_MISSING_CODE {
                        table.observe(code);
                    }
                }
            }
        }
        _ => {
            return Err(representation_changed("unknown"));
        }
    }
    Ok(())
}

/// A representation changed between streaming batches.
fn representation_changed(representation: &'static str) -> SKError {
    SKError::UnsupportedRepresentation {
        representation,
        suggestion: "stream batches of a single categorical representation",
    }
}
