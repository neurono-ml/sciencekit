//! Shareable per-pass reader factory and schema probe for Arrow file sources
//! (spec `arrow-columnar-interop`).

use std::path::{Path, PathBuf};

use arrow_schema::Schema;
use sciencekit_common::SKError;

use super::csv_reader::{sk_csv_schema, sk_open_csv_reader};
use super::format::SKArrowFileFormat;
use super::ipc_reader::{
    sk_ipc_file_schema, sk_ipc_stream_schema, sk_open_ipc_file_reader, sk_open_ipc_stream_reader,
};
use super::json_reader::{sk_json_schema, sk_open_json_reader};
use super::options::SKArrowFileOptions;
use super::parquet_reader::{sk_open_parquet_reader, sk_parquet_schema};

pub(super) fn sk_read_file_schema(
    path: &Path,
    format: SKArrowFileFormat,
    options: &SKArrowFileOptions,
) -> Result<Schema, SKError> {
    if options.explicit_schema.is_some()
        && matches!(
            format,
            SKArrowFileFormat::Parquet | SKArrowFileFormat::IpcFile | SKArrowFileFormat::IpcStream
        )
    {
        return Err(SKError::UnsupportedOperation {
            operation: "open_arrow_file_source",
            reason: format!(
                "{format} embeds its own schema; explicit schemas apply only to csv, json and ndjson sources"
            ),
        });
    }
    match format {
        SKArrowFileFormat::Parquet => sk_parquet_schema(path),
        SKArrowFileFormat::Csv => sk_csv_schema(path, options),
        SKArrowFileFormat::IpcFile => sk_ipc_file_schema(path),
        SKArrowFileFormat::IpcStream => sk_ipc_stream_schema(path),
        SKArrowFileFormat::Json | SKArrowFileFormat::NdJson => sk_json_schema(path, options),
    }
}

/// The plain, shareable state a reader factory captures once the file schema
/// and column projection have been resolved.
#[derive(Debug, Clone)]
pub(super) struct SKArrowFileOpener {
    pub(super) path: PathBuf,
    pub(super) format: SKArrowFileFormat,
    pub(super) options: SKArrowFileOptions,
    pub(super) file_schema: Schema,
    pub(super) projection: Vec<usize>,
}

impl SKArrowFileOpener {
    pub(super) fn open(&self) -> Result<Box<dyn arrow_array::RecordBatchReader + Send>, SKError> {
        match self.format {
            SKArrowFileFormat::Parquet => sk_open_parquet_reader(
                &self.path,
                self.options.batch_size,
                &self.file_schema,
                &self.projection,
            ),
            SKArrowFileFormat::Csv => sk_open_csv_reader(
                &self.path,
                &self.file_schema,
                &self.options,
                &self.projection,
            ),
            SKArrowFileFormat::IpcFile => {
                sk_open_ipc_file_reader(&self.path, &self.file_schema, &self.projection)
            }
            SKArrowFileFormat::IpcStream => {
                sk_open_ipc_stream_reader(&self.path, &self.file_schema, &self.projection)
            }
            SKArrowFileFormat::Json | SKArrowFileFormat::NdJson => sk_open_json_reader(
                &self.path,
                &self.file_schema,
                &self.options,
                &self.projection,
            ),
        }
    }
}
