//! A reader wrapper that reports the projected schema and optionally reorders
//! or projects each decoded batch (spec `arrow-columnar-interop`).
//!
//! Arrow's IPC readers keep the file schema on `RecordBatchReader::schema()`
//! even when a projection is applied, Parquet's `ProjectionMask::roots` yields
//! projected columns in ascending file order rather than selection order, and
//! CSV/JSON decode full records before projection. This wrapper normalizes all
//! four so every format reader yields batches matching its declared schema.

use std::sync::Arc;

use arrow_array::{RecordBatch, RecordBatchReader};
use arrow_schema::{ArrowError, Schema, SchemaRef};

/// Build the Arrow schema carrying only `indices` from `schema`, in that order.
pub(crate) fn sk_projected_schema(schema: &Schema, indices: &[usize]) -> Schema {
    Schema::new_with_metadata(
        indices
            .iter()
            .filter_map(|index| {
                schema
                    .fields()
                    .get(*index)
                    .map(|field| field.as_ref().clone())
            })
            .collect::<Vec<_>>(),
        schema.metadata().clone(),
    )
}

/// A reader with an explicitly declared schema and optional per-batch column
/// projection.
pub(crate) struct SKArrowProjectedReader {
    inner: Box<dyn RecordBatchReader + Send>,
    schema: SchemaRef,
    batch_projection: Option<Vec<usize>>,
}

impl SKArrowProjectedReader {
    /// Wrap a reader, declare the schema its batches carry, and optionally
    /// project/reorder each batch onto `batch_projection`.
    pub(crate) fn new(
        inner: Box<dyn RecordBatchReader + Send>,
        schema: Schema,
        batch_projection: Option<Vec<usize>>,
    ) -> Self {
        SKArrowProjectedReader {
            inner,
            schema: Arc::new(schema),
            batch_projection,
        }
    }
}

impl Iterator for SKArrowProjectedReader {
    type Item = Result<RecordBatch, ArrowError>;

    fn next(&mut self) -> Option<Self::Item> {
        let next_batch = self.inner.next()?;
        match (next_batch, &self.batch_projection) {
            (Ok(batch), Some(indices)) => Some(batch.project(indices)),
            (other, None) => Some(other),
            (Err(error), Some(_)) => Some(Err(error)),
        }
    }
}

impl RecordBatchReader for SKArrowProjectedReader {
    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }
}
