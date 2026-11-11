//! CSV file source with inferred or explicit schema and post-decode projection
//! (spec `arrow-columnar-interop`).

use std::fs::File;
use std::io::{Seek, SeekFrom};
use std::path::Path;
use std::sync::Arc;

use arrow_array::RecordBatchReader;
use arrow_csv::reader::{Format, ReaderBuilder};
use arrow_schema::Schema;
use sciencekit_common::SKError;

use super::options::SKArrowFileOptions;
use super::projected_reader::{SKArrowProjectedReader, sk_projected_schema};

fn sk_csv_error(path: &Path, error: impl std::fmt::Display) -> SKError {
    SKError::Conversion(format!(
        "failed to read CSV file `{}`: {error}",
        path.display()
    ))
}

/// Resolve the full CSV schema from an explicit override or header/row
/// inference.
pub(crate) fn sk_csv_schema(path: &Path, options: &SKArrowFileOptions) -> Result<Schema, SKError> {
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

/// Open a CSV file as an Arrow reader and project decoded batches onto the
/// resolved column indices.
///
/// `arrow-csv` decodes rows eagerly against the file schema; when a column
/// selection is active, [`SKArrowProjectedReader`] drops unselected columns and
/// reorders the remaining ones to match `projection`.
///
/// # Errors
/// Returns [`SKError::Io`] for an unreadable file and
/// [`SKError::Conversion`] when the file is malformed.
pub(crate) fn sk_open_csv_reader(
    path: &Path,
    file_schema: &Schema,
    options: &SKArrowFileOptions,
    projection: &[usize],
) -> Result<Box<dyn RecordBatchReader + Send>, SKError> {
    let file = File::open(path)?;
    let reader = ReaderBuilder::new(Arc::new(file_schema.clone()))
        .with_header(options.has_csv_header)
        .with_delimiter(options.csv_delimiter)
        .with_batch_size(options.batch_size)
        .build(file)
        .map_err(|error| sk_csv_error(path, error))?;
    if options.selection.is_none() {
        return Ok(Box::new(reader));
    }
    let projected_schema = sk_projected_schema(file_schema, projection);
    Ok(Box::new(SKArrowProjectedReader::new(
        Box::new(reader),
        projected_schema,
        Some(projection.to_vec()),
    )))
}
