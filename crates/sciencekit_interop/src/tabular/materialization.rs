//! Materialization of a neutral batch into a tabular dataset (spec
//! `tabular-sources`).

use ndarray::Array2;
use sciencekit_common::{SKError, SKFeatureBlock, SKFloat};

use super::{SKColumnKind, SKNominalMatrixOwned, SKTabularDataset, SKTabularSourceConfig};
use crate::batch::SKTabularBatch;
use crate::schema::{SKColumnSchema, SKTabularSchema};
use crate::selection::SKColumnSelection;

impl SKTabularBatch {
    /// Materialize the configured columns into an owned dataset.
    ///
    /// Numeric and boolean columns form the feature block in declaration
    /// order; categorical columns form the owned categorical block; ignored
    /// columns contribute nothing. Every null is resolved by the configured
    /// missing-value policy.
    ///
    /// # Errors
    /// Returns [`SKError::UnknownColumn`] for a configured name the batch does
    /// not carry, [`SKError::UnsupportedColumnType`] when a column's logical
    /// type cannot carry its declared kind, and propagates the numeric and
    /// categorical conversion failures.
    pub fn to_tabular_dataset<F: SKFloat>(
        &self,
        config: &SKTabularSourceConfig,
    ) -> Result<SKTabularDataset<F>, SKError> {
        let projected = self.project_configuration(config)?;

        let feature_selection = sk_selection_of_kind(config, |kind| kind.is_numeric())?;
        let features = match feature_selection {
            Some(selection) => self.to_feature_block::<F>(
                &selection,
                config.missing_value_policy(),
                config.layout(),
            )?,
            None => SKFeatureBlock::Dense(Array2::zeros((self.number_of_rows(), 0))),
        };

        let categorical_selection = sk_selection_of_kind(config, |kind| kind.is_categorical())?;
        let categorical = match categorical_selection {
            Some(selection) => Some(SKNominalMatrixOwned::from_extraction(
                self.extract_categorical(&selection, config.missing_value_policy())?,
            )?),
            None => None,
        };

        SKTabularDataset::new(projected, features, categorical)
    }

    fn project_configuration(
        &self,
        config: &SKTabularSourceConfig,
    ) -> Result<SKTabularSchema, SKError> {
        let mut columns: Vec<SKColumnSchema> = Vec::new();
        for (name, kind) in config.columns() {
            if *kind == SKColumnKind::Ignored {
                // An ignored column may have been dropped by the adapter; it
                // contributes nothing either way.
                continue;
            }
            let index = self
                .schema()
                .column_index(name)
                .ok_or_else(|| SKError::UnknownColumn {
                    column: name.clone(),
                    suggestion: format!(
                        "the source carries: {}",
                        self.schema()
                            .columns()
                            .iter()
                            .map(SKColumnSchema::name)
                            .collect::<Vec<_>>()
                            .join(", ")
                    ),
                })?;
            let column = &self.schema().columns()[index];
            if !kind.accepts(column.logical_type()) {
                return Err(SKError::UnsupportedColumnType {
                    column: name.clone(),
                    logical_type: column.logical_type().to_string(),
                    suggestion: format!(
                        "a `{kind}` column must carry a compatible logical type; \
                         use `ignored` or convert the column in the adapter"
                    ),
                });
            }
            columns.push(column.clone());
        }
        SKTabularSchema::new(columns)
    }
}

fn sk_selection_of_kind(
    config: &SKTabularSourceConfig,
    predicate: impl Fn(SKColumnKind) -> bool,
) -> Result<Option<SKColumnSelection>, SKError> {
    let names = config.column_names_of_kind(predicate);
    if names.is_empty() {
        return Ok(None);
    }
    let selection = names
        .into_iter()
        .fold(SKColumnSelection::builder(), |builder, name| {
            builder.feature(name)
        })
        .build()?;
    Ok(Some(selection))
}
