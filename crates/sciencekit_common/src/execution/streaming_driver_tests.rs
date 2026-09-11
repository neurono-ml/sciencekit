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

// ---- Task 4.3 scenarios (spec `streaming-executor`, control flow) ----

/// Early stop via the callback halts the pipeline: batch k's effect is kept, no
/// batch beyond the prefetched k+1 is requested, and the call returns without
/// error.
#[test]
fn early_stop_halts_the_pipeline() {
    let (mut source, read_log, _outstanding, _max) = RecordingSource::new(8, None);
    let update = |batch: SKDataBatch<f64>, _parallelism: usize, state: &mut Vec<usize>| {
        state.push(batch.position());
        if batch.position() == 2 {
            SKStreamDecision::Stop
        } else {
            SKStreamDecision::Continue
        }
    };
    let mut state = Vec::new();
    let result = sk_run_streaming_driver(&mut source, &mut state, update, &streaming_plan());

    assert!(result.is_ok(), "early stop should return without error");
    assert_eq!(state, vec![0, 1, 2]);
    let max_read = read_log
        .lock()
        .unwrap()
        .iter()
        .map(|(index, _)| *index)
        .max()
        .unwrap();
    assert!(
        max_read <= 3,
        "read beyond the prefetched batch: up to index {max_read}"
    );
}

/// A mid-stream read failure surfaces the central taxonomy error with the
/// effects of earlier batches preserved, and no further batch is processed.
#[test]
fn intermediate_read_failure_propagates_structured_error() {
    let (mut source, _read_log, _outstanding, _max) = RecordingSource::new(8, Some(3));
    let update = |batch: SKDataBatch<f64>, _parallelism: usize, state: &mut Vec<usize>| {
        state.push(batch.position());
        SKStreamDecision::Continue
    };
    let mut state = Vec::new();
    let result = sk_run_streaming_driver(&mut source, &mut state, update, &streaming_plan());

    match result {
        Err(SKError::Conversion(message)) => {
            assert!(message.contains("mid-stream read failure"));
        }
        other => panic!("expected taxonomy error, got {other:?}"),
    }
    assert_eq!(state, vec![0, 1, 2], "earlier batch effects must be preserved");
}

/// On a finite source, exactly one batch is delivered flagged final, and it is
/// the last one; the driver then terminates normally.
#[test]
fn final_batch_is_delivered_exactly_once() {
    let (mut source, _read_log, _outstanding, _max) = RecordingSource::new(5, None);
    let update = |batch: SKDataBatch<f64>, _parallelism: usize, state: &mut Vec<(usize, bool)>| {
        state.push((batch.position(), batch.is_final()));
        SKStreamDecision::Continue
    };
    let mut state = Vec::new();
    sk_run_streaming_driver(&mut source, &mut state, update, &streaming_plan()).unwrap();

    assert_eq!(state.len(), 5);
    let final_count = state.iter().filter(|(_, is_final)| *is_final).count();
    assert_eq!(final_count, 1);
    assert_eq!(state.last(), Some(&(4, true)));
}

// ---- Task 5.1 scenarios (PRD §8.7 acceptance: concurrency, metrics, sizes) ----

/// A streaming consumer accumulates a metric (here the total sum) across batches
/// and the driver returns success, exercising the "produces metrics" acceptance.
#[test]
fn streaming_consumer_produces_correct_metric() {
    let (mut source, _read_log, _outstanding, _max) = RecordingSource::new(6, None);
    let update = |batch: SKDataBatch<f64>, _parallelism: usize, state: &mut f64| {
        *state += batch.data()[(0, 0)];
        SKStreamDecision::Continue
    };
    let mut metric = 0.0;
    sk_run_streaming_driver(&mut source, &mut metric, update, &streaming_plan()).unwrap();
    let expected: f64 = (0..6).map(|i| i as f64).sum();
    assert_eq!(metric, expected);
}

/// The driver runs correctly from several threads concurrently, each with its
/// own source, delivering ordered batches everywhere.
#[test]
fn driver_runs_concurrently_across_threads() {
    let handles: Vec<_> = (0..4)
        .map(|_| {
            std::thread::spawn(|| {
                let (mut source, _read_log, _outstanding, _max) = RecordingSource::new(5, None);
                let update = |batch: SKDataBatch<f64>, _parallelism: usize, state: &mut Vec<usize>| {
                    state.push(batch.position());
                    SKStreamDecision::Continue
                };
                let mut state = Vec::new();
                sk_run_streaming_driver(&mut source, &mut state, update, &streaming_plan()).unwrap();
                state
            })
        })
        .collect();
    for handle in handles {
        assert_eq!(handle.join().unwrap(), vec![0, 1, 2, 3, 4]);
    }
}

/// The driver accepts both a single-batch (little data) and a 1000-batch (lots
/// of data) stream, and the sequential fallback (`parallelism = 1`) works
/// end-to-end.
#[test]
fn driver_accepts_small_and_large_streams_with_sequential_fallback() {
    let (mut small, _read_log, _outstanding, _max) = RecordingSource::new(1, None);
    let update = |batch: SKDataBatch<f64>, _parallelism: usize, state: &mut Vec<usize>| {
        state.push(batch.position());
        SKStreamDecision::Continue
    };
    let mut small_state = Vec::new();
    sk_run_streaming_driver(&mut small, &mut small_state, update, &streaming_plan()).unwrap();
    assert_eq!(small_state, vec![0]);

    let (mut large, _read_log, _outstanding, _max) = RecordingSource::new(1000, None);
    let sequential_plan = SKExecutionPlan {
        parallelism: 1,
        ..streaming_plan()
    };
    let mut large_state = Vec::new();
    sk_run_streaming_driver(&mut large, &mut large_state, update, &sequential_plan).unwrap();
    assert_eq!(large_state.len(), 1000);
    assert_eq!(*large_state.first().unwrap(), 0);
    assert_eq!(*large_state.last().unwrap(), 999);
}