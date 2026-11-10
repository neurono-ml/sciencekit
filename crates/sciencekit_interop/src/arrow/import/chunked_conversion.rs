//! Import of chunked columns (the table-like representation) into the neutral
//! seam (spec `arrow-columnar-interop`).
//!
//! Arrow's Rust implementation represents a chunked column as a
//! `Vec<ArrayRef>`: each chunk is one array and the column's logical values are
//! the chunks concatenated in order. This module turns that representation into
//! a single record batch and reuses the record-batch conversion.

use arrow_array::{ArrayRef, RecordBatch, new_empty_array};
use arrow_schema::{DataType, Schema};
use arrow_select::concat::concat;
use sciencekit_common::SKError;

use super::record_batch_conversion::sk_tabular_batch_from_record_batch;
use crate::batch::SKTabularBatch;
use crate::missing::SKMissingValuePolicy;

fn sk_chunk_conversion_error(message: String) -> SKError {
    SKError::Conversion(format!("Arrow chunked-column import failed: {message}"))
}

fn sk_chunk_schema_mismatch(expected: String, found: String) -> SKError {
    SKError::SchemaMismatch { expected, found }
}

/// Concatenate one column's chunks in order into a single array.
fn sk_concat_chunks(
    field_name: &str,
    data_type: &DataType,
    chunks: &[ArrayRef],
) -> Result<ArrayRef, SKError> {
    if chunks.is_empty() {
        return Ok(new_empty_array(data_type));
    }
    for chunk in chunks {
        if chunk.data_type() != data_type {
            return Err(sk_chunk_schema_mismatch(
                format!("column `{field_name}` of type {data_type}"),
                format!("a chunk of type {}", chunk.data_type()),
            ));
        }
    }
    if chunks.len() == 1 {
        return Ok(chunks[0].clone());
    }
    let borrowed: Vec<&dyn arrow_array::Array> =
        chunks.iter().map(|chunk| chunk.as_ref()).collect();
    concat(&borrowed).map_err(|error| sk_chunk_conversion_error(error.to_string()))
}

/// Convert a schema plus one chunk sequence per column into a tabular batch.
///
/// Chunks are concatenated in declaration order, so the imported column
/// preserves both chunk order and total length. A column with zero chunks
/// becomes a typed empty column instead of an error.
///
/// # Errors
/// Returns [`SKError::SchemaMismatch`] when the chunk count differs from the
/// schema's column count or a chunk carries a different data type, and
/// [`SKError::Conversion`] when Arrow's concatenation fails.
pub fn sk_tabular_batch_from_column_chunks(
    schema: &Schema,
    column_chunks: &[Vec<ArrayRef>],
    policy: SKMissingValuePolicy,
    position: usize,
    is_final: bool,
) -> Result<SKTabularBatch, SKError> {
    if column_chunks.len() != schema.fields().len() {
        return Err(sk_chunk_schema_mismatch(
            format!("{} columns", schema.fields().len()),
            format!("{} chunked columns", column_chunks.len()),
        ));
    }
    let columns = schema
        .fields()
        .iter()
        .zip(column_chunks)
        .map(|(field, chunks)| sk_concat_chunks(field.name(), field.data_type(), chunks))
        .collect::<Result<Vec<_>, SKError>>()?;
    let record_batch = RecordBatch::try_new(schema.clone().into(), columns)
        .map_err(|error| sk_chunk_conversion_error(error.to_string()))?;
    sk_tabular_batch_from_record_batch(&record_batch, policy, position, is_final)
}
