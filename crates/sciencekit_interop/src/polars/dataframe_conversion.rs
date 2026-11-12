//! Conversion of a Polars `DataFrame` into a neutral batch or a materialized
//! dataset (spec `tabular-sources`).

use ::polars::prelude::DataFrame;
use sciencekit_common::{SKError, SKFloat};

use super::column_conversion::sk_column_data_from_polars_series;
use super::dtype_mapping::sk_logical_type_from_polars_data_type;
use crate::batch::SKTabularBatch;
use crate::schema::{SKColumnRole, SKColumnSchema, SKTabularSchema};
use crate::tabular::{SKColumnKind, SKTabularDataset, SKTabularSourceConfig};

/// Convert a Polars `DataFrame` into a neutral batch under a column
/// configuration.
///
/// Columns appear in configuration order; ignored columns are dropped. The
/// batch preserves nulls as `None`, so the configured missing-value policy is
/// applied when the batch is materialized.
///
/// # Errors
/// Returns [`SKError::UnknownColumn`] for a configured name the frame does not
/// carry, [`SKError::UnsupportedColumnType`] for a dtype with no neutral
/// counterpart and the Polars conversion failures.
pub fn sk_tabular_batch_from_dataframe(
    frame: &DataFrame,
    config: &SKTabularSourceConfig,
    position: usize,
    is_final: bool,
) -> Result<SKTabularBatch, SKError> {
    let selected = sk_selected_columns(config, |kind| kind != SKColumnKind::Ignored);
    let mut schema_columns = Vec::with_capacity(selected.len());
    let mut columns = Vec::with_capacity(selected.len());
    for (name, _) in selected {
        let column = frame.column(name.as_str()).map_err(|_| SKError::UnknownColumn {
            column: name.clone(),
            suggestion: format!(
                "the frame carries: {}",
                frame.get_column_names_str().join(", ")
            ),
        })?;
        let series = column.as_materialized_series();
        let logical_type = sk_logical_type_from_polars_data_type(series.dtype(), &name)?;
        schema_columns.push(SKColumnSchema::new(
            name.clone(),
            logical_type,
            SKColumnRole::Feature,
        ));
        columns.push(sk_column_data_from_polars_series(
            &name,
            series,
            logical_type,
        )?);
    }
    let schema = SKTabularSchema::new(schema_columns)?;
    SKTabularBatch::new(schema, columns, position, is_final)
}

/// Convert a Polars `DataFrame` into an owned materialized dataset.
///
/// # Errors
/// Propagates [`sk_tabular_batch_from_dataframe`] and the materialization
/// failures of the configured policy.
pub fn sk_tabular_dataset_from_dataframe<F: SKFloat>(
    frame: &DataFrame,
    config: &SKTabularSourceConfig,
) -> Result<SKTabularDataset<F>, SKError> {
    sk_tabular_batch_from_dataframe(frame, config, 0, true)?.to_tabular_dataset::<F>(config)
}

impl<F: SKFloat> TryFrom<(&DataFrame, &SKTabularSourceConfig)> for SKTabularDataset<F> {
    type Error = SKError;

    fn try_from(
        (frame, config): (&DataFrame, &SKTabularSourceConfig),
    ) -> Result<Self, Self::Error> {
        sk_tabular_dataset_from_dataframe(frame, config)
    }
}

pub(crate) fn sk_selected_columns(
    config: &SKTabularSourceConfig,
    predicate: impl Fn(SKColumnKind) -> bool,
) -> Vec<(String, SKColumnKind)> {
    config
        .columns()
        .iter()
        .filter(|(_, kind)| predicate(*kind))
        .cloned()
        .collect()
}
