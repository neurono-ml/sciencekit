//! JSON and NDJSON sources (spec `arrow-columnar-interop`).
//!
//! JSON readers decode whole documents; projection is applied by the source
//! cursor after decoding. An explicit schema is required unless inference is
//! opted in, because guessing types from untyped JSON silently changes a
//! column's meaning.

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

fn sk_json_error(path: &Path, error: impl std::fmt::Display) -> SKError {
    SKError::Conversion(format!(
        "failed to read JSON file `{}`: {error}",
        path.display()
    ))
}

fn sk_json_schema(path: &Path, options: &SKArrowFileOptions) -> Result<Schema, SKError> {
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

/// Open a JSON array document or an NDJSON file as an Arrow reader.
///
/// # Errors
/// Returns [`SKError::UnsupportedOperation`] when no schema is supplied and
/// inference is off, [`SKError::Io`] for an unreadable file and
/// [`SKError::Conversion`] for a malformed document.
pub(crate) fn sk_open_json_reader(
    path: &Path,
    options: &SKArrowFileOptions,
) -> Result<Box<dyn RecordBatchReader + Send>, SKError> {
    let schema = sk_json_schema(path, options)?;
    let reader = BufReader::new(File::open(path)?);
    let json_reader = ReaderBuilder::new(Arc::new(schema))
        .with_batch_size(options.batch_size)
        .build(reader)
        .map_err(|error| sk_json_error(path, error))?;
    Ok(Box::new(json_reader))
}
