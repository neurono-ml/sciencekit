//! The supported columnar file formats (spec `arrow-columnar-interop`).

use std::fmt;

/// A file format the adapter can open as a tabular source.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SKArrowFileFormat {
    /// Apache Parquet.
    Parquet,
    /// Delimited text with an inferred or explicit schema.
    Csv,
    /// Arrow IPC file format (Feather v2).
    IpcFile,
    /// Arrow IPC stream format.
    IpcStream,
    /// Record-oriented JSON (a sequence of objects).
    Json,
    /// Newline-delimited JSON.
    NdJson,
}

impl SKArrowFileFormat {
    /// The stable, lower-case format name used in errors and traces.
    pub fn name(&self) -> &'static str {
        match self {
            SKArrowFileFormat::Parquet => "parquet",
            SKArrowFileFormat::Csv => "csv",
            SKArrowFileFormat::IpcFile => "ipc_file",
            SKArrowFileFormat::IpcStream => "ipc_stream",
            SKArrowFileFormat::Json => "json",
            SKArrowFileFormat::NdJson => "ndjson",
        }
    }
}

impl fmt::Display for SKArrowFileFormat {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.name())
    }
}
