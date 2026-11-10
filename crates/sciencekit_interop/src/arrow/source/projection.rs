//! Column projection for Arrow sources (spec `arrow-columnar-interop`).
//!
//! A source resolves the caller's selection once per pass and requests only
//! those columns from the reader, in selection order, followed by the target
//! column when the selection is supervised.

use arrow_array::RecordBatch;
use arrow_schema::Schema;
use sciencekit_common::SKError;

use super::super::schema_mapping::{SKArrowColumnRoles, sk_tabular_schema_from_arrow_schema};
use crate::selection::SKColumnSelection;

/// Resolve the selection into the Arrow column positions to read, in order.
///
/// With no selection every column is read in source order.
///
/// # Errors
/// Propagates the schema rejection for an unsupported column and the selection
/// errors for an unknown or duplicated column.
pub(crate) fn sk_projection_indices(
    schema: &Schema,
    selection: Option<&SKColumnSelection>,
) -> Result<Vec<usize>, SKError> {
    let Some(selection) = selection else {
        return Ok((0..schema.fields().len()).collect());
    };
    let neutral = sk_tabular_schema_from_arrow_schema(schema, &SKArrowColumnRoles::new())?;
    let resolved = selection.resolve(&neutral)?;
    let mut indices = resolved.feature_indices().to_vec();
    if let Some(target) = resolved.target_index() {
        if !indices.contains(&target) {
            indices.push(target);
        }
    }
    Ok(indices)
}

/// Project a record batch onto the resolved positions, preserving their order.
///
/// # Errors
/// Returns [`SKError::Conversion`] when Arrow rejects the projection, which can
/// only happen for an out-of-range position the resolution already excluded.
pub(crate) fn sk_project_record_batch(
    batch: &RecordBatch,
    indices: &[usize],
) -> Result<RecordBatch, SKError> {
    batch
        .project(indices)
        .map_err(|error| SKError::Conversion(format!("Arrow projection failed: {error}")))
}
