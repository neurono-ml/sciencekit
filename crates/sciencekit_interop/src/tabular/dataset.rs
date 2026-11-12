//! A materialized tabular dataset: schema, feature block and categorical
//! block (spec `tabular-sources`).

use sciencekit_common::{SKDataView, SKError, SKFeatureBlock, SKFloat};

use super::{SKColumnKind, SKNominalMatrixOwned};
use crate::schema::{SKLogicalType, SKTabularSchema};

/// A materialized table ready to feed an estimator.
///
/// The dataset pairs the projected schema with a dense or sparse numeric
/// feature block and an optional owned categorical block. It owns every buffer,
/// so it can be moved to another thread and read concurrently without touching
/// the engine that produced it.
#[derive(Debug, Clone)]
pub struct SKTabularDataset<F: SKFloat> {
    schema: SKTabularSchema,
    features: SKFeatureBlock<F>,
    categorical: Option<SKNominalMatrixOwned>,
}

impl<F: SKFloat> SKTabularDataset<F> {
    /// Assemble a dataset from its parts.
    ///
    /// # Errors
    /// Returns [`SKError::UnevenColumnLengths`] when the numeric and
    /// categorical blocks disagree on the row count, and
    /// [`SKError::SchemaMismatch`] when the projected schema's width differs, in
    /// either direction, from the blocks it describes.
    pub fn new(
        schema: SKTabularSchema,
        features: SKFeatureBlock<F>,
        categorical: Option<SKNominalMatrixOwned>,
    ) -> Result<Self, SKError> {
        let number_of_rows = features.number_of_rows();
        if let Some(categorical) = &categorical {
            if categorical.number_of_rows() != number_of_rows {
                return Err(SKError::UnevenColumnLengths {
                    column: "categorical".to_string(),
                    expected: number_of_rows,
                    found: categorical.number_of_rows(),
                });
            }
        }
        let number_of_feature_columns = features.number_of_columns();
        let number_of_categorical_columns = categorical
            .as_ref()
            .map_or(0, SKNominalMatrixOwned::number_of_columns);
        let expected_width = number_of_feature_columns + number_of_categorical_columns;
        if schema.len() != expected_width {
            return Err(SKError::SchemaMismatch {
                expected: format!("{expected_width} selected columns"),
                found: format!("{} schema columns", schema.len()),
            });
        }
        Ok(SKTabularDataset {
            schema,
            features,
            categorical,
        })
    }

    /// The projected schema of the non-ignored columns, in declaration order.
    pub fn schema(&self) -> &SKTabularSchema {
        &self.schema
    }

    /// The numeric feature block.
    pub fn features(&self) -> &SKFeatureBlock<F> {
        &self.features
    }

    /// Borrow the numeric features as a zero-copy data view.
    pub fn feature_view(&self) -> SKDataView<'_, F> {
        self.features.view()
    }

    /// The categorical block, when at least one categorical column was
    /// selected.
    pub fn categorical(&self) -> Option<&SKNominalMatrixOwned> {
        self.categorical.as_ref()
    }

    /// The kind of a projected column, derived from its logical type.
    ///
    /// Ignored columns never reach the projected schema, so a name present
    /// here always has a materialized kind.
    pub fn kind_of(&self, name: &str) -> Option<SKColumnKind> {
        let column = self.schema.column(name)?;
        Some(match column.logical_type() {
            SKLogicalType::Nominal | SKLogicalType::CategoryCode => SKColumnKind::Categorical,
            SKLogicalType::Boolean => SKColumnKind::Boolean,
            _ => SKColumnKind::Numeric,
        })
    }

    /// The number of rows.
    pub fn number_of_rows(&self) -> usize {
        self.features.number_of_rows()
    }

    /// The number of numeric feature columns.
    pub fn number_of_feature_columns(&self) -> usize {
        self.features.number_of_columns()
    }

    /// The number of categorical columns.
    pub fn number_of_categorical_columns(&self) -> usize {
        self.categorical
            .as_ref()
            .map_or(0, SKNominalMatrixOwned::number_of_columns)
    }

    /// Consume the dataset into its three parts.
    pub fn into_parts(
        self,
    ) -> (
        SKTabularSchema,
        SKFeatureBlock<F>,
        Option<SKNominalMatrixOwned>,
    ) {
        (self.schema, self.features, self.categorical)
    }
}
