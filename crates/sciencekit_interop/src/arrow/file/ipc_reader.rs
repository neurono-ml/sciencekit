//! Arrow IPC file and stream sources with projection pushdown (spec
//! `arrow-columnar-interop`).

use std::fs::File;
use std::path::Path;

use arrow_array::RecordBatchReader;
use arrow_ipc::reader::{FileReader, StreamReader};
use sciencekit_common::SKError;

use super::options::SKArrowFileOptions;
use crate::arrow::source::sk_projection_indices;

fn sk_ipc_error(path: &Path, error: impl std::fmt::Display) -> SKError {
    SKError::Conversion(format!(
        "failed to read Arrow IPC file `{}`: {error}",
        path.display()
    ))
}

/// Open an Arrow IPC file (Feather v2) as an Arrow reader.
///
/// The file is opened once to read the schema and again with the resolved
/// projection, so unselected columns are never decoded.
///
/// # Errors
/// Returns [`SKError::Io`] for an unreadable file and
/// [`SKError::Conversion`] for an invalid or truncated IPC file.
pub(crate) fn sk_open_ipc_file_reader(
    path: &Path,
    options: &SKArrowFileOptions,
) -> Result<Box<dyn RecordBatchReader + Send>, SKError> {
    let probe =
        FileReader::try_new(File::open(path)?, None).map_err(|error| sk_ipc_error(path, error))?;
    let indices = sk_projection_indices(probe.schema().as_ref(), options.selection.as_ref())?;
    let reader = FileReader::try_new(File::open(path)?, Some(indices))
        .map_err(|error| sk_ipc_error(path, error))?;
    Ok(Box::new(reader))
}

/// Open an Arrow IPC stream as an Arrow reader.
///
/// # Errors
/// Returns [`SKError::Io`] for an unreadable file and
/// [`SKError::Conversion`] for an invalid or truncated IPC stream.
pub(crate) fn sk_open_ipc_stream_reader(
    path: &Path,
    options: &SKArrowFileOptions,
) -> Result<Box<dyn RecordBatchReader + Send>, SKError> {
    let probe = StreamReader::try_new(File::open(path)?, None)
        .map_err(|error| sk_ipc_error(path, error))?;
    let indices = sk_projection_indices(probe.schema().as_ref(), options.selection.as_ref())?;
    let reader = StreamReader::try_new(File::open(path)?, Some(indices))
        .map_err(|error| sk_ipc_error(path, error))?;
    Ok(Box::new(reader))
}
