//! A reader wrapper that reports the projected schema (spec
//! `arrow-columnar-interop`).
//!
//! Arrow's IPC readers keep the file schema even when a projection is applied,
//! while their batches carry only the projected columns. The source cursor
//! resolves its selection against the reader's schema, so the wrapper exposes
//! the schema the batches actually carry.

use std::sync::Arc;

use arrow_array::{RecordBatch, RecordBatchReader};
use arrow_schema::{ArrowError, Schema, SchemaRef};

/// A reader with an explicitly declared, projected schema.
pub(crate) struct SKArrowProjectedReader {
    inner: Box<dyn RecordBatchReader + Send>,
    schema: SchemaRef,
}

impl SKArrowProjectedReader {
    /// Wrap a reader and declare the schema its batches carry.
    pub(crate) fn new(inner: Box<dyn RecordBatchReader + Send>, schema: Schema) -> Self {
        SKArrowProjectedReader {
            inner,
            schema: Arc::new(schema),
        }
    }
}

impl Iterator for SKArrowProjectedReader {
    type Item = Result<RecordBatch, ArrowError>;

    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next()
    }
}

impl RecordBatchReader for SKArrowProjectedReader {
    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }
}
