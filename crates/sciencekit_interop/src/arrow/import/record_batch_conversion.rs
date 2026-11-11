//! Conversion of an Arrow record batch into a neutral tabular batch (spec
//! `arrow-columnar-interop`).

use arrow_array::RecordBatch;
use sciencekit_common::SKError;

use super::super::schema_mapping::{SKArrowColumnRoles, sk_tabular_schema_from_arrow_schema};
use super::column_conversion::sk_column_data_from_arrow_array;
use crate::batch::SKTabularBatch;
use crate::columns::SKColumnData;
use crate::missing::SKMissingValuePolicy;
use crate::observability::{
    SKInteropConversionAttributes, SKInteropSourceKind, sk_run_interop_conversion,
};
use crate::schema::SKLogicalType;

pub(super) fn sk_convert_record_batch_without_span(
    record_batch: &RecordBatch,
    roles: &SKArrowColumnRoles,
    policy: SKMissingValuePolicy,
    position: usize,
    is_final: bool,
) -> Result<SKTabularBatch, SKError> {
    let schema = sk_tabular_schema_from_arrow_schema(record_batch.schema_ref(), roles)?;
    let columns = schema
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
    SKTabularBatch::new(schema, columns, position, is_final)
}

/// Convert a record batch into a tabular batch with explicit column-role
/// assignments, preserving column names, order, logical types and row count.
///
/// Field metadata (`sciencekit:role`) still takes precedence over `roles`,
/// matching [`sk_tabular_schema_from_arrow_schema`]. The conversion is
/// instrumented under a `sciencekit.interop.conversion` span with
/// [`SKInteropSourceKind::Arrow`].
///
/// # Errors
/// Propagates the schema rejection for an unsupported column type and the
/// policy errors for a null or fill value the policy rejects.
pub fn sk_tabular_batch_from_record_batch_with_roles(
    record_batch: &RecordBatch,
    roles: &SKArrowColumnRoles,
    policy: SKMissingValuePolicy,
    position: usize,
    is_final: bool,
) -> Result<SKTabularBatch, SKError> {
    let attributes = SKInteropConversionAttributes {
        source_kind: SKInteropSourceKind::Arrow,
        rows_count: record_batch.num_rows(),
        columns_count: record_batch.num_columns(),
        missing_value_policy: policy,
    };
    sk_run_interop_conversion(attributes, || {
        sk_convert_record_batch_without_span(record_batch, roles, policy, position, is_final)
    })
}

/// Convert a record batch into a tabular batch, preserving column names, order,
/// logical types and row count.
///
/// Roles come from `sciencekit:role` field metadata when present and default to
/// [`crate::schema::SKColumnRole::Feature`] otherwise; callers that want to
/// assign roles without mutating the Arrow schema use
/// [`sk_tabular_batch_from_record_batch_with_roles`].
///
/// # Errors
/// Propagates the schema rejection for an unsupported column type and the
/// policy errors for a null or fill value the policy rejects.
pub fn sk_tabular_batch_from_record_batch(
    record_batch: &RecordBatch,
    policy: SKMissingValuePolicy,
    position: usize,
    is_final: bool,
) -> Result<SKTabularBatch, SKError> {
    sk_tabular_batch_from_record_batch_with_roles(
        record_batch,
        &SKArrowColumnRoles::new(),
        policy,
        position,
        is_final,
    )
}
