//! Arrow IPC file and stream sources with projection pushdown (spec
//! `arrow-columnar-interop`).

use std::fs::File;
use std::path::Path;

use arrow_array::RecordBatchReader;
use arrow_ipc::reader::{FileReader, StreamReader};
use arrow_schema::Schema;
use sciencekit_common::SKError;

use super::projected_reader::{SKArrowProjectedReader, sk_projected_schema};

fn sk_ipc_error(path: &Path, error: impl std::fmt::Display) -> SKError {
    SKError::Conversion(format!(
        "failed to read Arrow IPC file `{}`: {error}",
        path.display()
    ))
}

/// Read the unprojected Arrow schema of an Arrow IPC file (Feather v2).
pub(crate) fn sk_ipc_file_schema(path: &Path) -> Result<Schema, SKError> {
    let probe =
        FileReader::try_new(File::open(path)?, None).map_err(|error| sk_ipc_error(path, error))?;
    Ok(probe.schema().as_ref().clone())
}

/// Read the unprojected Arrow schema of an Arrow IPC stream.
pub(crate) fn sk_ipc_stream_schema(path: &Path) -> Result<Schema, SKError> {
    let probe = StreamReader::try_new(File::open(path)?, None)
        .map_err(|error| sk_ipc_error(path, error))?;
    Ok(probe.schema().as_ref().clone())
}

/// Open an Arrow IPC file (Feather v2) with the resolved column projection, so
/// unselected columns are never decoded and selection order is preserved.
///
/// # Errors
/// Returns [`SKError::Io`] for an unreadable file and
/// [`SKError::Conversion`] for an invalid or truncated IPC file.
pub(crate) fn sk_open_ipc_file_reader(
    path: &Path,
    file_schema: &Schema,
    projection: &[usize],
) -> Result<Box<dyn RecordBatchReader + Send>, SKError> {
    let projected_schema = sk_projected_schema(file_schema, projection);
    let reader = FileReader::try_new(File::open(path)?, Some(projection.to_vec()))
        .map_err(|error| sk_ipc_error(path, error))?;
    Ok(Box::new(SKArrowProjectedReader::new(
        Box::new(reader),
        projected_schema,
        None,
    )))
}

/// Open an Arrow IPC stream with the resolved column projection, so unselected
/// columns are never decoded and selection order is preserved.
///
/// # Errors
/// Returns [`SKError::Io`] for an unreadable file and
/// [`SKError::Conversion`] for an invalid or truncated IPC stream.
pub(crate) fn sk_open_ipc_stream_reader(
    path: &Path,
    file_schema: &Schema,
    projection: &[usize],
) -> Result<Box<dyn RecordBatchReader + Send>, SKError> {
    let projected_schema = sk_projected_schema(file_schema, projection);
    let reader = StreamReader::try_new(File::open(path)?, Some(projection.to_vec()))
        .map_err(|error| sk_ipc_error(path, error))?;
    Ok(Box::new(SKArrowProjectedReader::new(
        Box::new(reader),
        projected_schema,
        None,
    )))
}
