//! A one-shot tabular source over a single Arrow record-batch reader (spec
//! `arrow-columnar-interop`).

use arrow_array::RecordBatchReader;
use sciencekit_common::SKError;

use super::cursor::SKArrowBatchCursor;
use crate::batch::SKTabularBatch;
use crate::missing::SKMissingValuePolicy;
use crate::selection::SKColumnSelection;
use crate::source::SKTabularSource;

/// A source that consumes one Arrow reader exactly once.
///
/// Used for readers that cannot be recreated (a live stream, a socket-backed
/// reader). Iterative estimators that need several epochs require a reopenable
/// [`super::SKArrowRecordBatchSource`].
pub struct SKArrowOneShotTabularSource {
    reader: Option<Box<dyn RecordBatchReader + Send>>,
    selection: Option<SKColumnSelection>,
    batch_size: usize,
    policy: SKMissingValuePolicy,
    cursor: Option<SKArrowBatchCursor>,
    opened: bool,
}

impl SKArrowOneShotTabularSource {
    /// Wrap one Arrow reader.
    ///
    /// # Errors
    /// Returns [`SKError::InvalidHyperparameter`] when `batch_size` is zero.
    pub fn new(
        reader: Box<dyn RecordBatchReader + Send>,
        selection: Option<SKColumnSelection>,
        batch_size: usize,
        policy: SKMissingValuePolicy,
    ) -> Result<Self, SKError> {
        if batch_size == 0 {
            return Err(SKError::InvalidHyperparameter {
                name: "batch_size",
                reason: "a streaming batch must hold at least one row".to_string(),
            });
        }
        Ok(SKArrowOneShotTabularSource {
            reader: Some(reader),
            selection,
            batch_size,
            policy,
            cursor: None,
            opened: false,
        })
    }
}

impl SKTabularSource for SKArrowOneShotTabularSource {
    const IS_REOPENABLE: bool = false;

    fn open(&mut self) -> Result<(), SKError> {
        if self.opened {
            return Err(SKError::UnsupportedOperation {
                operation: "open_arrow_one_shot_source",
                reason: "the reader was already consumed and cannot be reopened".to_string(),
            });
        }
        let reader = self
            .reader
            .take()
            .ok_or_else(|| SKError::UnsupportedOperation {
                operation: "open_arrow_one_shot_source",
                reason: "the reader was already consumed and cannot be reopened".to_string(),
            })?;
        let schema = reader.schema();
        self.cursor = Some(SKArrowBatchCursor::new(
            reader,
            schema.as_ref(),
            self.selection.as_ref(),
            self.batch_size,
            self.policy,
        )?);
        self.opened = true;
        Ok(())
    }

    fn next_batch(&mut self) -> Result<Option<SKTabularBatch>, SKError> {
        if !self.opened {
            return Err(SKError::UnsupportedOperation {
                operation: "arrow_source_next_batch",
                reason: "the source must be opened before reading".to_string(),
            });
        }
        match self.cursor.as_mut() {
            Some(cursor) => cursor.next_batch(),
            None => Ok(None),
        }
    }
}
