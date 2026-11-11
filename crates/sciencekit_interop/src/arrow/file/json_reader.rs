//! Record-oriented JSON and NDJSON sources (spec `arrow-columnar-interop`).
//!
//! Arrow's JSON decoder reads record-oriented JSON: one object per line or a
//! concatenated sequence of objects, which covers NDJSON and JSON-lines
//! exports. Projection is applied after decoding onto the resolved selection
//! indices. An explicit schema is required unless inference is opted in,
//! because guessing types from untyped JSON silently changes a column's
//! meaning.

use std::fs::File;
use std::io::BufReader;
use std::path::Path;
use std::sync::Arc;

use arrow_array::RecordBatchReader;
use arrow_json::ReaderBuilder;
use arrow_json::reader::infer_json_schema_from_seekable;
use arrow_schema::Schema;
use sciencekit_common::SKError;

use super::options::SKArrowFileOptions;
use super::projected_reader::{SKArrowProjectedReader, sk_projected_schema};

fn sk_json_error(path: &Path, error: impl std::fmt::Display) -> SKError {
    SKError::Conversion(format!(
        "failed to read JSON file `{}`: {error}",
        path.display()
    ))
}

/// Resolve the full JSON schema from an explicit declaration or opted-in
/// inference.
pub(crate) fn sk_json_schema(path: &Path, options: &SKArrowFileOptions) -> Result<Schema, SKError> {
    if let Some(schema) = &options.explicit_schema {
        return Ok(schema.clone());
    }
    if !options.infer_json_schema {
        return Err(SKError::UnsupportedOperation {
            operation: "open_json_source",
            reason: "JSON has no schema; supply an explicit schema or enable inference".to_string(),
        });
    }
    let reader = BufReader::new(File::open(path)?);
    let (schema, _) = infer_json_schema_from_seekable(reader, Some(1_000))
        .map_err(|error| sk_json_error(path, error))?;
    Ok(schema)
}

/// Open a record-oriented JSON or NDJSON file as an Arrow reader and project
/// decoded batches onto the resolved column indices.
///
/// # Errors
/// Returns [`SKError::Io`] for an unreadable file and
/// [`SKError::Conversion`] for a malformed document.
pub(crate) fn sk_open_json_reader(
    path: &Path,
    file_schema: &Schema,
    options: &SKArrowFileOptions,
    projection: &[usize],
) -> Result<Box<dyn RecordBatchReader + Send>, SKError> {
    let reader = BufReader::new(File::open(path)?);
    let json_reader = ReaderBuilder::new(Arc::new(file_schema.clone()))
        .with_batch_size(options.batch_size)
        .build(reader)
        .map_err(|error| sk_json_error(path, error))?;
    if options.selection.is_none() {
        return Ok(Box::new(json_reader));
    }
    let projected_schema = sk_projected_schema(file_schema, projection);
    Ok(Box::new(SKArrowProjectedReader::new(
        Box::new(json_reader),
        projected_schema,
        Some(projection.to_vec()),
    )))
}
