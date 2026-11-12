//! A reopenable streaming source over a DataFusion `DataFrame` (spec
//! `tabular-sources`).

use ::datafusion::arrow::record_batch::RecordBatch;
use ::datafusion::dataframe::DataFrame;
use ::datafusion::physical_plan::SendableRecordBatchStream;
use futures::StreamExt;
use sciencekit_common::SKError;
use tokio::runtime::Runtime;

use super::batch_conversion::sk_tabular_batch_from_datafusion_record_batch;
use super::dataframe_conversion::sk_datafusion_runtime;
use crate::batch::SKTabularBatch;
use crate::source::SKTabularSource;
use crate::tabular::SKTabularSourceConfig;

/// Reject a call made from inside a Tokio runtime.
///
/// The source owns a runtime and blocks on it per batch, which Tokio forbids to
/// nest. Returning a structured error keeps the public surface panic-free for a
/// caller that happens to hold the source inside an async task.
fn sk_reject_nested_runtime(operation: &'static str) -> Result<(), SKError> {
    if tokio::runtime::Handle::try_current().is_ok() {
        return Err(SKError::UnsupportedOperation {
            operation,
            reason:
                "the streaming source blocks on its own runtime; drive it outside a Tokio runtime"
                    .to_string(),
        });
    }
    Ok(())
}

/// A reopenable source that pulls DataFusion record batches one at a time.
///
/// The physical plan runs as a stream: each [`next_batch`](SKTabularSource::next_batch)
/// polls the next record batch and converts it, so the adapter never collects
/// the whole result set. One batch is buffered ahead only to mark the final
/// block, which keeps row order and the `is_final` contract intact.
pub struct SKDataFusionStreamSource {
    dataframe: DataFrame,
    config: SKTabularSourceConfig,
    runtime: Runtime,
    stream: Option<SendableRecordBatchStream>,
    pending: Option<RecordBatch>,
    position: usize,
    opened: bool,
}

impl SKDataFusionStreamSource {
    /// Build a source over an unexecuted DataFusion plan.
    ///
    /// # Errors
    /// Returns [`SKError::Conversion`] when the Tokio runtime cannot be built
    /// and [`SKError::UnsupportedOperation`] when called inside a Tokio runtime,
    /// where the source's own runtime could neither be driven nor dropped.
    pub fn new(dataframe: DataFrame, config: SKTabularSourceConfig) -> Result<Self, SKError> {
        sk_reject_nested_runtime("datafusion_stream_source_new")?;
        Ok(SKDataFusionStreamSource {
            dataframe,
            config,
            runtime: sk_datafusion_runtime()?,
            stream: None,
            pending: None,
            position: 0,
            opened: false,
        })
    }

    fn pull(&mut self) -> Result<Option<RecordBatch>, SKError> {
        let stream = self
            .stream
            .as_mut()
            .ok_or_else(|| SKError::UnsupportedOperation {
                operation: "datafusion_stream_next_batch",
                reason: "the source must be opened before reading".to_string(),
            })?;
        match self.runtime.block_on(stream.next()) {
            Some(Ok(record_batch)) => Ok(Some(record_batch)),
            Some(Err(error)) => Err(SKError::Conversion(format!(
                "DataFusion stream failed: {error}"
            ))),
            None => Ok(None),
        }
    }
}

impl SKTabularSource for SKDataFusionStreamSource {
    const IS_REOPENABLE: bool = true;

    fn open(&mut self) -> Result<(), SKError> {
        sk_reject_nested_runtime("datafusion_stream_open")?;
        let stream = self
            .runtime
            .block_on(self.dataframe.clone().execute_stream())
            .map_err(|error| {
                SKError::Conversion(format!("DataFusion execute_stream failed: {error}"))
            })?;
        self.stream = Some(stream);
        self.pending = None;
        self.position = 0;
        self.opened = true;
        Ok(())
    }

    fn next_batch(&mut self) -> Result<Option<SKTabularBatch>, SKError> {
        sk_reject_nested_runtime("datafusion_stream_next_batch")?;
        if !self.opened {
            return Err(SKError::UnsupportedOperation {
                operation: "datafusion_stream_next_batch",
                reason: "the source must be opened before reading".to_string(),
            });
        }
        let record_batch = match self.pending.take() {
            Some(record_batch) => record_batch,
            None => match self.pull()? {
                Some(record_batch) => record_batch,
                None => return Ok(None),
            },
        };
        let next = self.pull()?;
        let is_final = next.is_none();
        self.pending = next;

        let position = self.position;
        self.position += 1;
        sk_tabular_batch_from_datafusion_record_batch(
            &record_batch,
            &self.config,
            position,
            is_final,
        )
        .map(Some)
    }
}
