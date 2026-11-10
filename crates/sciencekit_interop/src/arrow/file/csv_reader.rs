//! CSV file source with inferred or explicit schema and projection (spec
//! `arrow-columnar-interop`).

use std::fs::File;
use std::io::{Seek, SeekFrom};
use std::path::Path;
use std::sync::Arc;

use arrow_array::RecordBatchReader;
use arrow_csv::reader::{Format, ReaderBuilder};
use arrow_schema::Schema;
use sciencekit_common::SKError;

use super::options::SKArrowFileOptions;
use crate::arrow::source::sk_projection_indices;

fn sk_csv_error(path: &Path, error: impl std::fmt::Display) -> SKError {
    SKError::Conversion(format!(
        "failed to read CSV file `{}`: {error}",
        path.display()
    ))
}

fn sk_csv_schema(path: &Path, options: &SKArrowFileOptions) -> Result<Schema, SKError> {
    if let Some(schema) = &options.explicit_schema {
        return Ok(schema.clone());
    }
    let mut file = File::open(path)?;
    let format = Format::default()
        .with_header(options.has_csv_header)
        .with_delimiter(options.csv_delimiter);
    let (schema, _) = format
        .infer_schema(&mut file, Some(1_000))
        .map_err(|error| sk_csv_error(path, error))?;
    file.seek(SeekFrom::Start(0))?;
    Ok(schema)
}

/// Open a CSV file as an Arrow reader, reading only the selected columns.
///
/// # Errors
/// Returns [`SKError::Io`] for an unreadable file and
/// [`SKError::Conversion`] when the schema cannot be inferred or the file is
/// malformed.
pub(crate) fn sk_open_csv_reader(
    path: &Path,
    options: &SKArrowFileOptions,
) -> Result<Box<dyn RecordBatchReader + Send>, SKError> {
    let schema = sk_csv_schema(path, options)?;
    let indices = sk_projection_indices(&schema, options.selection.as_ref())?;
    let file = File::open(path)?;
    let reader = ReaderBuilder::new(Arc::new(schema))
        .with_header(options.has_csv_header)
        .with_delimiter(options.csv_delimiter)
        .with_batch_size(options.batch_size)
        .with_projection(indices)
        .build(file)
        .map_err(|error| sk_csv_error(path, error))?;
    Ok(Box::new(reader))
}
