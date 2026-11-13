//! The async-to-sync bridge: a dedicated reader thread with a bounded channel
//! (spec `dataset-interop-core`).

use std::future::poll_fn;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, SyncSender, sync_channel};
use std::sync::{Arc, Mutex};
use std::task::{Poll, Waker};
use std::thread::{self, JoinHandle, ThreadId};

use sciencekit_common::SKError;

use super::SKAsyncTabularStream;
use crate::batch::SKTabularBatch;
use crate::source::SKTabularSource;

/// The default buffer depth: double buffering, one batch being produced and one
/// queued.
pub const SK_DEFAULT_ASYNC_BRIDGE_DEPTH: usize = 1;

/// A synchronous [`SKTabularSource`] fed by an asynchronous stream.
///
/// A dedicated reader thread builds a current-thread Tokio runtime, polls the
/// stream there and sends owned batches through a bounded channel. Backpressure
/// bounds the resident batches to `depth + 1`; dropping the source closes the
/// channel, cancels the stream and joins the thread.
pub struct SKAsyncTabularSource {
    receiver: Option<Receiver<Result<SKTabularBatch, SKError>>>,
    reader_thread: Option<JoinHandle<()>>,
    cancellation: Arc<AtomicBool>,
    waker: Arc<Mutex<Option<Waker>>>,
    opened: bool,
    reader_identifier: ThreadId,
}

impl SKAsyncTabularSource {
    /// Start the reader thread and return the synchronous source.
    ///
    /// `depth` is the bounded channel capacity; zero is treated as one, the
    /// double-buffering default.
    pub fn new(stream: impl SKAsyncTabularStream + 'static, depth: usize) -> Self {
        let depth = depth.max(SK_DEFAULT_ASYNC_BRIDGE_DEPTH);
        let (sender, receiver) = sync_channel(depth);
        let cancellation = Arc::new(AtomicBool::new(false));
        let waker: Arc<Mutex<Option<Waker>>> = Arc::new(Mutex::new(None));

        let reader_cancellation = Arc::clone(&cancellation);
        let reader_waker = Arc::clone(&waker);
        let reader_thread = thread::Builder::new()
            .name("sciencekit-interop-reader".to_string())
            .spawn(move || {
                sk_run_reader(stream, sender, reader_cancellation, reader_waker);
            })
            .expect("the interop reader thread must be spawnable");
        let reader_identifier = reader_thread.thread().id();

        SKAsyncTabularSource {
            receiver: Some(receiver),
            reader_thread: Some(reader_thread),
            cancellation,
            waker,
            opened: false,
            reader_identifier,
        }
    }

    /// The identifier of the thread polling the asynchronous stream.
    pub fn reader_thread_identifier(&self) -> ThreadId {
        self.reader_identifier
    }
}

fn sk_run_reader(
    mut stream: impl SKAsyncTabularStream,
    sender: SyncSender<Result<SKTabularBatch, SKError>>,
    cancellation: Arc<AtomicBool>,
    waker: Arc<Mutex<Option<Waker>>>,
) {
    let runtime = match tokio::runtime::Builder::new_current_thread().build() {
        Ok(runtime) => runtime,
        Err(error) => {
            let _ = sender.send(Err(SKError::Io(std::io::Error::other(format!(
                "failed to build the interop reader runtime: {error}"
            )))));
            return;
        }
    };

    runtime.block_on(async move {
        loop {
            let item = poll_fn(|context| {
                *waker.lock().expect("the waker slot is not poisoned") =
                    Some(context.waker().clone());
                if cancellation.load(Ordering::SeqCst) {
                    return Poll::Ready(None);
                }
                stream.poll_next_batch(context)
            })
            .await;

            match item {
                Some(Ok(batch)) => {
                    if sender.send(Ok(batch)).is_err() {
                        break;
                    }
                }
                Some(Err(error)) => {
                    let _ = sender.send(Err(error));
                    break;
                }
                None => break,
            }
        }
    });
}

impl SKTabularSource for SKAsyncTabularSource {
    fn open(&mut self) -> Result<(), SKError> {
        if self.opened {
            return Err(SKError::UnsupportedOperation {
                operation: "open_async_tabular_source",
                reason: "an async bridge is one-shot and was already opened".to_string(),
            });
        }
        self.opened = true;
        Ok(())
    }

    fn next_batch(&mut self) -> Result<Option<SKTabularBatch>, SKError> {
        let Some(receiver) = &self.receiver else {
            return Ok(None);
        };
        match receiver.recv() {
            Ok(Ok(batch)) => Ok(Some(batch)),
            Ok(Err(error)) => Err(error),
            Err(_) => Ok(None),
        }
    }
}

impl Drop for SKAsyncTabularSource {
    fn drop(&mut self) {
        // Close the channel first so a reader blocked on `send` unblocks.
        self.receiver.take();
        self.cancellation.store(true, Ordering::SeqCst);
        if let Some(waker) = self
            .waker
            .lock()
            .expect("the waker slot is not poisoned")
            .take()
        {
            waker.wake();
        }
        if let Some(reader_thread) = self.reader_thread.take() {
            let _ = reader_thread.join();
        }
    }
}
