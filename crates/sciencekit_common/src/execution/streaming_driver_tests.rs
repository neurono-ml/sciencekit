//! Tests for the streaming executor driver (spec `streaming-executor`).

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use ndarray::Array2;

use super::streaming_driver::{SKStreamDecision, sk_run_streaming_driver};
use super::{SKExecutionMode, SKExecutionPlan};
use crate::{SKDataBatch, SKError, SKLazySource};

/// An instrumented sequential source: logs every read and can fail mid-stream.
struct RecordingSource {
    next: usize,
    total: usize,
    fail_at: Option<usize>,
    read_log: Arc<Mutex<Vec<(usize, Instant)>>>,
    outstanding: Arc<AtomicUsize>,
    max_outstanding: Arc<AtomicUsize>,
}

impl RecordingSource {
    fn new(total: usize, fail_at: Option<usize>) -> (Self, Arc<Mutex<Vec<(usize, Instant)>>>, Arc<AtomicUsize>, Arc<AtomicUsize>) {
        let read_log = Arc::new(Mutex::new(Vec::new()));
        let outstanding = Arc::new(AtomicUsize::new(0));
        let max_outstanding = Arc::new(AtomicUsize::new(0));
        let source = RecordingSource {
            next: 0,
            total,
            fail_at,
            read_log: Arc::clone(&read_log),
            outstanding: Arc::clone(&outstanding),
            max_outstanding: Arc::clone(&max_outstanding),
        };
        (source, read_log, outstanding, max_outstanding)
    }
}

impl SKLazySource<f64> for RecordingSource {
    type Error = SKError;
    fn next_batch(&mut self) -> Result<Option<SKDataBatch<f64>>, Self::Error> {
        if let Some(index) = self.fail_at {
            if self.next == index {
                return Err(SKError::Conversion("mid-stream read failure".into()));
            }
        }
        if self.next >= self.total {
            return Ok(None);
        }
        let position = self.next;
        self.next += 1;
        self.read_log.lock().unwrap().push((position, Instant::now()));
        let current = self.outstanding.fetch_add(1, Ordering::SeqCst) + 1;
        self.max_outstanding
            .fetch_max(current, Ordering::SeqCst);
        let data = Array2::from_shape_vec((1, 1), vec![position as f64]).unwrap();
        Ok(Some(SKDataBatch::new(data, position, position == self.total - 1)))
    }
}

/// A double-buffered streaming plan.
fn streaming_plan() -> SKExecutionPlan {
    SKExecutionPlan {
        mode: SKExecutionMode::OutOfCoreStreaming,
        parallelism: 2,
        batch_size: Some(1),
        buffer_depth: 1,
    }
}

// ---- Task 4.1 scenarios (spec `streaming-executor`, prefetch contract) ----

/// The reader advances while compute runs: the source is asked for batch k+1
/// before batch k's computation completes.
#[test]
fn reader_advances_while_compute_runs() {
    let (mut source, read_log, _outstanding, _max) = RecordingSource::new(5, None);
    let finish_log: Arc<Mutex<Vec<(usize, Instant)>>> = Arc::new(Mutex::new(Vec::new()));
    let finish = Arc::clone(&finish_log);
    let update = move |batch: SKDataBatch<f64>, _parallelism: usize, _state: &mut ()| {
        thread::sleep(Duration::from_millis(50));
        finish.lock().unwrap().push((batch.position(), Instant::now()));
        SKStreamDecision::Continue
    };
    let mut state = ();
    sk_run_streaming_driver(&mut source, &mut state, update, &streaming_plan()).unwrap();

    let reads = read_log.lock().unwrap().clone();
    let finishes = finish_log.lock().unwrap().clone();
    let read_time = |index: usize| reads.iter().find(|(i, _)| *i == index).map(|(_, t)| *t);
    let finish_time = |index: usize| finishes.iter().find(|(i, _)| *i == index).map(|(_, t)| *t);

    // For every k except the last, batch k+1 was read before batch k finished.
    for k in 0..4 {
        let read_next = read_time(k + 1).expect("batch k+1 must be read");
        let done = finish_time(k).expect("batch k must complete");
        assert!(
            read_next < done,
            "batch {} read after batch {} finished — no prefetch overlap",
            k + 1,
            k
        );
    }
}

/// Ordered delivery of owned batches: each batch arrives in source order and its
/// data survives after the source has advanced.
#[test]
fn ordered_delivery_of_owned_batches() {
    let (mut source, _read_log, _outstanding, _max) = RecordingSource::new(5, None);
    let update = |batch: SKDataBatch<f64>, _parallelism: usize, state: &mut Vec<(usize, Vec<f64>)>| {
        let data = batch.data().as_slice_memory_order().unwrap().to_vec();
        state.push((batch.position(), data));
        SKStreamDecision::Continue
    };
    let mut state = Vec::new();
    sk_run_streaming_driver(&mut source, &mut state, update, &streaming_plan()).unwrap();

    assert_eq!(state.len(), 5);
    for (index, (position, data)) in state.iter().enumerate() {
        assert_eq!(*position, index);
        assert_eq!(data, &[index as f64]);
    }
}

/// Buffer depth is respected: at most two batches are resident at any moment
/// (the one computing plus the one prefetched).
#[test]
fn buffer_depth_is_respected() {
    let (mut source, _read_log, outstanding, max_outstanding) = RecordingSource::new(8, None);
    let outstanding_ref = Arc::clone(&outstanding);
    let update = move |_batch: SKDataBatch<f64>, _parallelism: usize, _state: &mut ()| {
        // Model compute time so the prefetched batch stays resident a moment.
        thread::sleep(Duration::from_millis(5));
        outstanding_ref.fetch_sub(1, Ordering::SeqCst);
        SKStreamDecision::Continue
    };
    let mut state = ();
    sk_run_streaming_driver(&mut source, &mut state, update, &streaming_plan()).unwrap();

    assert!(
        max_outstanding.load(Ordering::SeqCst) <= 2,
        "buffer depth violated: {} batches resident",
        max_outstanding.load(Ordering::SeqCst)
    );
}