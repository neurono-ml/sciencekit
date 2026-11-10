//! Conversion of an Arrow record batch into a neutral tabular batch (spec
//! `arrow-columnar-interop`).

use arrow_array::RecordBatch;
use sciencekit_common::SKError;

use super::super::schema_mapping::{SKArrowColumnRoles, sk_tabular_schema_from_arrow_schema};
use super::column_conversion::sk_column_data_from_arrow_array;
use crate::batch::SKTabularBatch;
use crate::columns::SKColumnData;
use crate::missing::SKMissingValuePolicy;
use crate::schema::{SKColumnSchema, SKLogicalType, SKTabularSchema};

/// Convert a record batch into a tabular batch, preserving column names, order
/// and row count.
///
/// Every Arrow column maps through [`sk_column_data_from_arrow_array`] under
/// the declared policy, and the resulting schema is the Arrow schema mapped
/// onto the neutral vocabulary. Sliced batches keep their logical window: the
/// array offsets are resolved by Arrow's own value accessors.
///
/// # Errors
/// Propagates the schema rejection for an unsupported column type and the
/// policy errors for a null the policy does not handle.
pub fn sk_tabular_batch_from_record_batch(
    record_batch: &RecordBatch,
    policy: SKMissingValuePolicy,
    position: usize,
    is_final: bool,
) -> Result<SKTabularBatch, SKError> {
    let base_schema =
        sk_tabular_schema_from_arrow_schema(record_batch.schema_ref(), &SKArrowColumnRoles::new())?;
    let columns = base_schema
        .columns()
        .iter()
        .zip(record_batch.columns())
        .map(|(column_schema, array)| {
            let column =
                sk_column_data_from_arrow_array(column_schema.name(), array.as_ref(), policy)?;
            // An exported category column travels as a plain Int64 with a
            // logical-type metadata declaration; restore its neutral type.
            Ok(match (column_schema.logical_type(), column) {
                (SKLogicalType::CategoryCode, SKColumnData::Integer64(values)) => {
                    SKColumnData::CategoryCode(values)
                }
                (_, column) => column,
            })
        })
        .collect::<Result<Vec<_>, SKError>>()?;
    // A nullable integer column promoted to float64 by the missing-value policy
    // changes its logical type; the schema must describe the data it carries.
    let promoted_columns = base_schema
        .columns()
        .iter()
        .zip(&columns)
        .map(|(column_schema, column)| {
            let declared = column_schema.logical_type();
            let logical_type = if declared.is_numeric()
                && column.logical_type().is_numeric()
                && declared != column.logical_type()
            {
                column.logical_type()
            } else {
                declared
            };
            SKColumnSchema::new(column_schema.name(), logical_type, column_schema.role())
        })
        .collect::<Vec<_>>();
    let schema = SKTabularSchema::new(promoted_columns)?;
    SKTabularBatch::new(schema, columns, position, is_final)
}
