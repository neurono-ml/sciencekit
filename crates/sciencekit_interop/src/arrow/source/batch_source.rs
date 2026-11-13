//! A reopenable tabular source over an Arrow record-batch reader factory
//! (spec `arrow-columnar-interop`).

use std::sync::Arc;

use arrow_array::RecordBatchReader;
use sciencekit_common::SKError;

use super::cursor::SKArrowBatchCursor;
use crate::batch::SKTabularBatch;
use crate::missing::SKMissingValuePolicy;
use crate::selection::SKColumnSelection;
use crate::source::SKTabularSource;

type SKArrowReaderFactory =
    Arc<dyn Fn() -> Result<Box<dyn RecordBatchReader + Send>, SKError> + Send + Sync>;

/// A source that recreates its Arrow reader on every pass.
///
/// The factory is the reopenability contract: a file path or an in-memory
/// batch supplier can be reopened, while a one-shot stream uses
/// [`super::SKArrowOneShotTabularSource`].
pub struct SKArrowRecordBatchSource {
    factory: SKArrowReaderFactory,
    selection: Option<SKColumnSelection>,
    batch_size: usize,
    policy: SKMissingValuePolicy,
    cursor: Option<SKArrowBatchCursor>,
    opened: bool,
}

impl SKArrowRecordBatchSource {
    /// Build a reopenable source from a reader factory.
    ///
    /// # Errors
    /// Returns [`SKError::InvalidHyperparameter`] when `batch_size` is zero.
    pub fn new<Factory>(
        factory: Factory,
        selection: Option<SKColumnSelection>,
        batch_size: usize,
        policy: SKMissingValuePolicy,
    ) -> Result<Self, SKError>
    where
        Factory: Fn() -> Result<Box<dyn RecordBatchReader + Send>, SKError> + Send + Sync + 'static,
    {
        if batch_size == 0 {
            return Err(SKError::InvalidHyperparameter {
                name: "batch_size",
                reason: "a streaming batch must hold at least one row".to_string(),
            });
        }
        Ok(SKArrowRecordBatchSource {
            factory: Arc::new(factory),
            selection,
            batch_size,
            policy,
            cursor: None,
            opened: false,
        })
    }

    /// The configured maximum number of rows per delivered batch.
    pub fn batch_size(&self) -> usize {
        self.batch_size
    }
}

impl SKTabularSource for SKArrowRecordBatchSource {
    const IS_REOPENABLE: bool = true;

    fn open(&mut self) -> Result<(), SKError> {
        let reader = (self.factory)()?;
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
