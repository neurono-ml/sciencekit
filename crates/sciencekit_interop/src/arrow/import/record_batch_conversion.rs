//! Conversion of an Arrow record batch into a neutral tabular batch (spec
//! `arrow-columnar-interop`).

use arrow_array::RecordBatch;
use sciencekit_common::SKError;

use super::super::schema_mapping::{SKArrowColumnRoles, sk_tabular_schema_from_arrow_schema};
use super::column_conversion::sk_column_data_from_arrow_array;
use crate::batch::SKTabularBatch;
use crate::missing::SKMissingValuePolicy;

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
    let schema =
        sk_tabular_schema_from_arrow_schema(record_batch.schema_ref(), &SKArrowColumnRoles::new())?;
    let columns = schema
        .columns()
        .iter()
        .zip(record_batch.columns())
        .map(|(column_schema, array)| {
            sk_column_data_from_arrow_array(column_schema.name(), array.as_ref(), policy)
        })
        .collect::<Result<Vec<_>, SKError>>()?;
    SKTabularBatch::new(schema, columns, position, is_final)
}
