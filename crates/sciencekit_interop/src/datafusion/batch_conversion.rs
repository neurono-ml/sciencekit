//! Conversion of a DataFusion Arrow record batch into a neutral batch (spec
//! `tabular-sources`).

use ::datafusion::arrow::record_batch::RecordBatch;
use sciencekit_common::SKError;

use super::column_conversion::sk_column_data_from_datafusion_array;
use super::dtype_mapping::sk_logical_type_from_datafusion_data_type;
use crate::batch::SKTabularBatch;
use crate::schema::{SKColumnRole, SKColumnSchema, SKTabularSchema};
use crate::tabular::SKColumnKind;
use crate::tabular::SKTabularSourceConfig;

/// Convert a DataFusion record batch into a neutral batch under a column
/// configuration.
///
/// Columns appear in configuration order; ignored columns are dropped and nulls
/// stay `None` until the configured policy is applied.
///
/// # Errors
/// Returns [`SKError::UnknownColumn`] for a configured name the batch does not
/// carry and [`SKError::UnsupportedColumnType`] for an unsupported dtype.
pub fn sk_tabular_batch_from_datafusion_record_batch(
    record_batch: &RecordBatch,
    config: &SKTabularSourceConfig,
    position: usize,
    is_final: bool,
) -> Result<SKTabularBatch, SKError> {
    let schema = record_batch.schema();
    let mut schema_columns = Vec::new();
    let mut columns = Vec::new();
    for (name, kind) in config.columns() {
        if *kind == SKColumnKind::Ignored {
            continue;
        }
        let index = schema.index_of(name).map_err(|_| SKError::UnknownColumn {
            column: name.clone(),
            suggestion: format!(
                "the batch carries: {}",
                schema
                    .fields()
                    .iter()
                    .map(|field| field.name().as_str())
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
        })?;
        let field = schema.field(index);
        let logical_type = sk_logical_type_from_datafusion_data_type(field.data_type(), name)?;
        schema_columns.push(SKColumnSchema::new(
            name.clone(),
            logical_type,
            SKColumnRole::Feature,
        ));
        columns.push(sk_column_data_from_datafusion_array(
            name,
            record_batch.column(index),
            logical_type,
        )?);
    }
    SKTabularBatch::new(
        SKTabularSchema::new(schema_columns)?,
        columns,
        position,
        is_final,
    )
}
