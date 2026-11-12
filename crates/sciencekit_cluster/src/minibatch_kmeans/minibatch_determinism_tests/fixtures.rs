//! Shared deterministic fixtures and span recorder for MiniBatchKMeans
//! determinism and concurrency tests.

use std::fmt;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

use ndarray::{Array2, Axis, Slice};
use sciencekit_common::{SKDataBatch, SKError, SKLazySource};
use tracing::field::{Field, Visit};
use tracing::span::{Attributes, Id, Record};

pub(super) fn assert_send_and_sync<T: Send + Sync>() {}

pub(super) fn deterministic_dataset(rows: usize) -> Array2<f64> {
    let mut matrix = Array2::<f64>::zeros((rows, 3));
    matrix
        .rows_mut()
        .into_iter()
        .enumerate()
        .for_each(|(row_index, mut row)| {
            let group = (row_index % 3) as f64 * 50.0;
            row[0] = group + ((row_index * 5) % 11) as f64 * 0.1;
            row[1] = group - ((row_index * 7) % 13) as f64 * 0.1;
            row[2] = group + ((row_index * 3) % 7) as f64 * 0.05;
        });
    matrix
}

pub(super) struct ChunkedSource {
    pub(super) data: Array2<f64>,
    pub(super) cursor: usize,
    pub(super) chunk_size: usize,
}

impl SKLazySource<f64> for ChunkedSource {
    type Error = SKError;

    fn next_batch(&mut self) -> Result<Option<SKDataBatch<f64>>, Self::Error> {
        if self.cursor >= self.data.nrows() {
            return Ok(None);
        }
        let end = (self.cursor + self.chunk_size).min(self.data.nrows());
        let block = self
            .data
            .slice_axis(Axis(0), Slice::from(self.cursor..end))
            .to_owned();
        let position = self.cursor / self.chunk_size;
        self.cursor = end;
        Ok(Some(SKDataBatch::new_dense(
            block,
            position,
            self.cursor >= self.data.nrows(),
        )))
    }
}

#[derive(Default)]
pub(super) struct BackendSpanRecorder {
    next_id: AtomicU64,
    pub(super) backends: Mutex<Vec<String>>,
}

struct BackendVisitor(Option<String>);

impl Visit for BackendVisitor {
    fn record_str(&mut self, field: &Field, value: &str) {
        if field.name() == "backend" {
            self.0 = Some(value.to_string());
        }
    }

    fn record_debug(&mut self, field: &Field, value: &dyn fmt::Debug) {
        if field.name() == "backend" {
            self.0 = Some(format!("{value:?}"));
        }
    }
}

impl tracing::Subscriber for BackendSpanRecorder {
    fn enabled(&self, _: &tracing::Metadata<'_>) -> bool {
        true
    }

    fn new_span(&self, span: &Attributes<'_>) -> Id {
        let mut visitor = BackendVisitor(None);
        span.values().record(&mut visitor);
        if let Some(backend) = visitor.0 {
            self.backends.lock().unwrap().push(backend);
        }
        Id::from_u64(self.next_id.fetch_add(1, Ordering::Relaxed) + 1)
    }

    fn record(&self, _: &Id, _: &Record<'_>) {}
    fn record_follows_from(&self, _: &Id, _: &Id) {}
    fn event(&self, _: &tracing::Event<'_>) {}
    fn enter(&self, _: &Id) {}
    fn exit(&self, _: &Id) {}
}
