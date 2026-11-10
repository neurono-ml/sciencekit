//! Parquet file source with row-group batching and projection pushdown (spec
//! `arrow-columnar-interop`).

use std::fs::File;
use std::path::Path;

use arrow_array::RecordBatchReader;
use parquet::arrow::ProjectionMask;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use sciencekit_common::SKError;

use super::options::SKArrowFileOptions;
use crate::arrow::source::sk_projection_indices;

/// Open a Parquet file as an Arrow reader, reading only the selected columns.
///
/// # Errors
/// Returns [`SKError::Io`] for an unreadable file and
/// [`SKError::Conversion`] for an invalid Parquet file or a rejected schema.
pub(crate) fn sk_open_parquet_reader(
    path: &Path,
    options: &SKArrowFileOptions,
) -> Result<Box<dyn RecordBatchReader + Send>, SKError> {
    let file = File::open(path)?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file).map_err(|error| {
        SKError::Conversion(format!(
            "failed to open Parquet file `{}`: {error}",
            path.display()
        ))
    })?;
    let schema = builder.schema().clone();
    let indices = sk_projection_indices(schema.as_ref(), options.selection.as_ref())?;
    let mask = ProjectionMask::roots(builder.parquet_schema(), indices);
    let reader = builder
        .with_batch_size(options.batch_size)
        .with_projection(mask)
        .build()
        .map_err(|error| {
            SKError::Conversion(format!(
                "failed to read Parquet file `{}`: {error}",
                path.display()
            ))
        })?;
    Ok(Box::new(reader))
}
