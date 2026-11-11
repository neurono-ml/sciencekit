//! Parquet file source with row-group batching and projection pushdown (spec
//! `arrow-columnar-interop`).

use std::fs::File;
use std::path::Path;

use arrow_array::RecordBatchReader;
use arrow_schema::Schema;
use parquet::arrow::ProjectionMask;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use sciencekit_common::SKError;

use super::projected_reader::{SKArrowProjectedReader, sk_projected_schema};

fn sk_parquet_builder(path: &Path) -> Result<ParquetRecordBatchReaderBuilder<File>, SKError> {
    let file = File::open(path)?;
    ParquetRecordBatchReaderBuilder::try_new(file).map_err(|error| {
        SKError::Conversion(format!(
            "failed to open Parquet file `{}`: {error}",
            path.display()
        ))
    })
}

/// Read the unprojected Arrow schema from the Parquet footer.
pub(crate) fn sk_parquet_schema(path: &Path) -> Result<Schema, SKError> {
    Ok(sk_parquet_builder(path)?.schema().as_ref().clone())
}

/// Open a Parquet file as an Arrow reader, decoding only `projection` columns
/// and reordering the decoded columns into `projection` order.
///
/// # Errors
/// Returns [`SKError::Io`] for an unreadable file and
/// [`SKError::Conversion`] for an invalid Parquet file.
pub(crate) fn sk_open_parquet_reader(
    path: &Path,
    batch_size: usize,
    file_schema: &Schema,
    projection: &[usize],
) -> Result<Box<dyn RecordBatchReader + Send>, SKError> {
    let builder = sk_parquet_builder(path)?;
    let mask = ProjectionMask::roots(builder.parquet_schema(), projection.iter().copied());
    let reader = builder
        .with_batch_size(batch_size)
        .with_projection(mask)
        .build()
        .map_err(|error| {
            SKError::Conversion(format!(
                "failed to read Parquet file `{}`: {error}",
                path.display()
            ))
        })?;

    // `ProjectionMask::roots` yields the selected columns in ascending file
    // order; compute the post-read permutation when the caller asked for a
    // different column order.
    let mut file_order = projection.to_vec();
    file_order.sort_unstable();
    let reorder: Vec<usize> = projection
        .iter()
        .map(|target| {
            file_order
                .binary_search(target)
                .expect("every projected index is in the sorted set")
        })
        .collect();
    let needs_reorder = reorder
        .iter()
        .enumerate()
        .any(|(position, &index)| position != index);
    let projected_schema = sk_projected_schema(file_schema, projection);
    Ok(Box::new(SKArrowProjectedReader::new(
        Box::new(reader),
        projected_schema,
        needs_reorder.then_some(reorder),
    )))
}
