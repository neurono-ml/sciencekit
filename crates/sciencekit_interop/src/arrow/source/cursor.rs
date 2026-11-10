//! The per-pass cursor over an Arrow record-batch reader (spec
//! `arrow-columnar-interop`).

use arrow_array::{RecordBatch, RecordBatchReader};
use arrow_schema::Schema;
use sciencekit_common::SKError;

use super::projection::{sk_project_record_batch, sk_projection_indices};
use crate::batch::SKTabularBatch;
use crate::missing::SKMissingValuePolicy;
use crate::selection::SKColumnSelection;

/// Splits a reader's record batches into tabular batches of at most
/// `batch_size` rows.
///
/// The cursor keeps one reader batch in flight and one buffered ahead, so it
/// knows whether the final delivered batch is truly the last one before it
/// marks it final.
pub(crate) struct SKArrowBatchCursor {
    reader: Box<dyn RecordBatchReader + Send>,
    projection: Vec<usize>,
    batch_size: usize,
    policy: SKMissingValuePolicy,
    position: usize,
    current: Option<RecordBatch>,
    current_offset: usize,
    buffered_next: Option<RecordBatch>,
    pending_error: Option<SKError>,
}

impl SKArrowBatchCursor {
    pub(crate) fn new(
        reader: Box<dyn RecordBatchReader + Send>,
        schema: &Schema,
        selection: Option<&SKColumnSelection>,
        batch_size: usize,
        policy: SKMissingValuePolicy,
    ) -> Result<Self, SKError> {
        if batch_size == 0 {
            return Err(SKError::InvalidHyperparameter {
                name: "batch_size",
                reason: "a streaming batch must hold at least one row".to_string(),
            });
        }
        let projection = sk_projection_indices(schema, selection)?;
        Ok(SKArrowBatchCursor {
            reader,
            projection,
            batch_size,
            policy,
            position: 0,
            current: None,
            current_offset: 0,
            buffered_next: None,
            pending_error: None,
        })
    }

    fn pull_projected(&mut self) -> Result<Option<RecordBatch>, SKError> {
        match self.reader.next() {
            None => Ok(None),
            Some(Ok(batch)) => Ok(Some(sk_project_record_batch(&batch, &self.projection)?)),
            Some(Err(error)) => Err(SKError::Conversion(format!(
                "Arrow record-batch reader failed: {error}"
            ))),
        }
    }

    pub(crate) fn next_batch(&mut self) -> Result<Option<SKTabularBatch>, SKError> {
        loop {
            if self.current.is_none() {
                self.current = self.buffered_next.take();
                if self.current.is_none() {
                    if let Some(error) = self.pending_error.take() {
                        return Err(error);
                    }
                    self.current = self.pull_projected()?;
                }
                self.current_offset = 0;
                if self.current.is_none() {
                    return Ok(None);
                }
            }

            let window = {
                let current = self.current.as_ref().expect("a current batch");
                let remaining = current.num_rows().saturating_sub(self.current_offset);
                if remaining == 0 {
                    None
                } else {
                    Some(current.slice(self.current_offset, remaining.min(self.batch_size)))
                }
            };
            let Some(window) = window else {
                self.current = None;
                continue;
            };

            if self.buffered_next.is_none() && self.pending_error.is_none() {
                // A lookahead failure must not swallow a complete window the
                // reader already produced: deliver it first, then surface the
                // error on the next call.
                match self.pull_projected() {
                    Ok(batch) => self.buffered_next = batch,
                    Err(error) => self.pending_error = Some(error),
                }
            }
            let current_rows = self.current.as_ref().map_or(0, RecordBatch::num_rows);
            self.current_offset += window.num_rows();
            let is_final = self.current_offset >= current_rows
                && self.buffered_next.is_none()
                && self.pending_error.is_none();
            let position = self.position;
            self.position += 1;
            let tabular = super::super::import::sk_tabular_batch_from_record_batch(
                &window,
                self.policy,
                position,
                is_final,
            )?;
            return Ok(Some(tabular));
        }
    }
}
