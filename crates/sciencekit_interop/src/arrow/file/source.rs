//! The path-based, reopenable Arrow file source (spec
//! `arrow-columnar-interop`).

use std::path::{Path, PathBuf};

use arrow_schema::Schema;
use sciencekit_common::SKError;

use super::format::SKArrowFileFormat;
use super::opener::{SKArrowFileOpener, sk_read_file_schema};
use super::options::SKArrowFileOptions;
use crate::arrow::source::sk_projection_indices;
use crate::batch::SKTabularBatch;
use crate::missing::SKMissingValuePolicy;
use crate::selection::SKColumnSelection;
use crate::source::SKTabularSource;

/// A reopenable source over one columnar or text file.
///
/// Every [`open`](SKTabularSource::open) creates a fresh reader from the path,
/// so iterative estimators receive the same batches in the same order on every
/// pass. Parquet and Arrow IPC push the column projection down into the format
/// reader so unselected columns are not decoded; CSV and record-oriented
/// JSON/NDJSON project the selected columns after record decoding.
pub struct SKArrowFileSource {
    path: PathBuf,
    format: SKArrowFileFormat,
    options: SKArrowFileOptions,
    inner: Option<crate::arrow::SKArrowRecordBatchSource>,
}

impl SKArrowFileSource {
    /// Open a file of the given format with the documented defaults.
    pub fn new(path: impl Into<PathBuf>, format: SKArrowFileFormat) -> Self {
        SKArrowFileSource {
            path: path.into(),
            format,
            options: SKArrowFileOptions::default(),
            inner: None,
        }
    }

    /// The file path.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// The file format.
    pub fn format(&self) -> SKArrowFileFormat {
        self.format
    }

    /// Set the maximum number of rows per delivered batch.
    pub fn with_batch_size(mut self, batch_size: usize) -> Self {
        self.options.batch_size = batch_size;
        self
    }

    /// Restrict reading to the selected columns, in selection order.
    pub fn with_selection(mut self, selection: SKColumnSelection) -> Self {
        self.options.selection = Some(selection);
        self
    }

    /// Set the missing-value policy applied to imported batches.
    pub fn with_missing_value_policy(mut self, policy: SKMissingValuePolicy) -> Self {
        self.options.policy = policy;
        self
    }

    /// Supply an explicit Arrow schema for text formats (`Csv`, `Json`,
    /// `NdJson`) instead of header/record inference.
    ///
    /// Self-describing binary formats (`Parquet`, `IpcFile`, `IpcStream`) carry
    /// their own schema and reject an explicit override at
    /// [`open`](SKTabularSource::open) with [`SKError::UnsupportedOperation`].
    pub fn with_explicit_schema(mut self, schema: Schema) -> Self {
        self.options.explicit_schema = Some(schema);
        self
    }

    /// Declare whether the CSV file carries a header row.
    pub fn with_csv_header(mut self, has_header: bool) -> Self {
        self.options.has_csv_header = has_header;
        self
    }

    /// Set the CSV field delimiter.
    pub fn with_csv_delimiter(mut self, delimiter: u8) -> Self {
        self.options.csv_delimiter = delimiter;
        self
    }

    /// Opt in to JSON schema inference when no explicit schema is supplied.
    pub fn with_json_schema_inference(mut self, infer: bool) -> Self {
        self.options.infer_json_schema = infer;
        self
    }
}

impl SKTabularSource for SKArrowFileSource {
    const IS_REOPENABLE: bool = true;

    fn open(&mut self) -> Result<(), SKError> {
        if self.options.batch_size == 0 {
            return Err(SKError::InvalidHyperparameter {
                name: "batch_size",
                reason: "a streaming batch must hold at least one row".to_string(),
            });
        }
        let file_schema = sk_read_file_schema(&self.path, self.format, &self.options)?;
        let projection = sk_projection_indices(&file_schema, self.options.selection.as_ref())?;
        let opener = SKArrowFileOpener {
            path: self.path.clone(),
            format: self.format,
            options: self.options.clone(),
            file_schema,
            projection,
        };
        let mut batch_source = crate::arrow::SKArrowRecordBatchSource::new(
            move || opener.open(),
            None,
            self.options.batch_size,
            self.options.policy,
        )?;
        batch_source.open()?;
        self.inner = Some(batch_source);
        Ok(())
    }

    fn next_batch(&mut self) -> Result<Option<SKTabularBatch>, SKError> {
        match self.inner.as_mut() {
            Some(inner) => inner.next_batch(),
            None => Err(SKError::UnsupportedOperation {
                operation: "arrow_file_source_next_batch",
                reason: "the source must be opened before reading".to_string(),
            }),
        }
    }
}
