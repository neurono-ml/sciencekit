//! Shared scripted async streams for the bridge tests.

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll};
use std::thread::ThreadId;

use sciencekit_common::SKError;

use super::SKAsyncTabularStream;
use crate::batch::SKTabularBatch;
use crate::source::test_fixtures::fixture_batch;

/// Observations recorded by a scripted stream.
#[derive(Debug, Clone, Default)]
pub(crate) struct StreamProbe {
    produced: Arc<AtomicUsize>,
    poll_threads: Arc<Mutex<Vec<ThreadId>>>,
    dropped: Arc<AtomicBool>,
}

impl StreamProbe {
    /// The number of batches the stream has produced.
    pub(crate) fn produced(&self) -> usize {
        self.produced.load(Ordering::SeqCst)
    }

    /// The identifiers of every thread that polled the stream.
    pub(crate) fn poll_threads(&self) -> Vec<ThreadId> {
        self.poll_threads.lock().unwrap().clone()
    }

    /// Whether the stream was dropped.
    pub(crate) fn was_dropped(&self) -> bool {
        self.dropped.load(Ordering::SeqCst)
    }
}

/// A finite scripted stream that yields one-row fixture batches.
pub(crate) struct ScriptedStream {
    total: usize,
    next: usize,
    fail_at: Option<usize>,
    probe: StreamProbe,
}

impl ScriptedStream {
    /// A stream of `total` batches.
    pub(crate) fn new(total: usize, probe: StreamProbe) -> Self {
        ScriptedStream {
            total,
            next: 0,
            fail_at: None,
            probe,
        }
    }

    /// A stream of `total` batches that fails with a conversion error at
    /// position `fail_at`.
    pub(crate) fn failing(total: usize, fail_at: usize, probe: StreamProbe) -> Self {
        ScriptedStream {
            total,
            next: 0,
            fail_at: Some(fail_at),
            probe,
        }
    }
}

impl SKAsyncTabularStream for ScriptedStream {
    fn poll_next_batch(
        &mut self,
        _context: &mut Context<'_>,
    ) -> Poll<Option<Result<SKTabularBatch, SKError>>> {
        self.probe
            .poll_threads
            .lock()
            .unwrap()
            .push(std::thread::current().id());

        if Some(self.next) == self.fail_at {
            return Poll::Ready(Some(Err(SKError::Conversion(
                "the engine stream failed".to_string(),
            ))));
        }
        if self.next >= self.total {
            return Poll::Ready(None);
        }
        let position = self.next;
        self.next += 1;
        self.probe.produced.fetch_add(1, Ordering::SeqCst);
        Poll::Ready(Some(Ok(fixture_batch(
            position,
            position + 1 == self.total,
        ))))
    }
}

impl Drop for ScriptedStream {
    fn drop(&mut self) {
        self.probe.dropped.store(true, Ordering::SeqCst);
    }
}
