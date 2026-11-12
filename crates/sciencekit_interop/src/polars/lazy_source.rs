//! A reopenable batch source over a Polars `LazyFrame` (spec
//! `tabular-sources`).

use ::polars::prelude::{DataFrame, LazyFrame};
use sciencekit_common::SKError;

use super::dataframe_conversion::sk_tabular_batch_from_dataframe;
use crate::batch::SKTabularBatch;
use crate::source::SKTabularSource;
use crate::tabular::SKTabularSourceConfig;

/// A reopenable source that turns a Polars `LazyFrame` into batches.
///
/// Polars 0.46 exposes no incremental sink over a `LazyFrame`: the query is
/// executed once per [`open`](SKTabularSource::open) and the resulting frame is
/// sliced into owned batches of at most `batch_size` rows. The batching keeps
/// downstream memory bounded per block and preserves row order; the execution
/// itself is the lazy engine's, not a row-by-row pull.
pub struct SKPolarsLazySource {
    lazy_frame: LazyFrame,
    config: SKTabularSourceConfig,
    batch_size: usize,
    batches: Vec<SKTabularBatch>,
    cursor: usize,
    opened: bool,
}

impl SKPolarsLazySource {
    /// Build a source from a lazy frame, a column configuration and a batch
    /// width.
    ///
    /// # Errors
    /// Returns [`SKError::InvalidHyperparameter`] when `batch_size` is zero.
    pub fn new(
        lazy_frame: LazyFrame,
        config: SKTabularSourceConfig,
        batch_size: usize,
    ) -> Result<Self, SKError> {
        if batch_size == 0 {
            return Err(SKError::InvalidHyperparameter {
                name: "batch_size",
                reason: "a streaming batch must hold at least one row".to_string(),
            });
        }
        Ok(SKPolarsLazySource {
            lazy_frame,
            config,
            batch_size,
            batches: Vec::new(),
            cursor: 0,
            opened: false,
        })
    }

    /// The configured maximum number of rows per delivered batch.
    pub fn batch_size(&self) -> usize {
        self.batch_size
    }
}

impl SKTabularSource for SKPolarsLazySource {
    const IS_REOPENABLE: bool = true;

    fn open(&mut self) -> Result<(), SKError> {
        let frame = self
            .lazy_frame
            .clone()
            .collect()
            .map_err(|error| SKError::Conversion(format!("Polars collect failed: {error}")))?;
        self.batches = sk_batches_from_dataframe(&frame, &self.config, self.batch_size)?;
        self.cursor = 0;
        self.opened = true;
        Ok(())
    }

    fn next_batch(&mut self) -> Result<Option<SKTabularBatch>, SKError> {
        if !self.opened {
            return Err(SKError::UnsupportedOperation {
                operation: "polars_lazy_source_next_batch",
                reason: "the source must be opened before reading".to_string(),
            });
        }
        let batch = self.batches.get(self.cursor).cloned();
        if batch.is_some() {
            self.cursor += 1;
        }
        Ok(batch)
    }
}

fn sk_batches_from_dataframe(
    frame: &DataFrame,
    config: &SKTabularSourceConfig,
    batch_size: usize,
) -> Result<Vec<SKTabularBatch>, SKError> {
    let height = frame.height();
    if height == 0 {
        return Ok(vec![sk_tabular_batch_from_dataframe(
            frame, config, 0, true,
        )?]);
    }
    let starts: Vec<usize> = (0..height).step_by(batch_size).collect();
    let number_of_batches = starts.len();
    starts
        .into_iter()
        .enumerate()
        .map(|(position, start)| {
            let length = usize::min(batch_size, height - start);
            let slice = frame.slice(start as i64, length);
            sk_tabular_batch_from_dataframe(
                &slice,
                config,
                position,
                position + 1 == number_of_batches,
            )
        })
        .collect()
}
