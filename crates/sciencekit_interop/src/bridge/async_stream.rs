//! The asynchronous batch stream the bridge adapts (spec
//! `dataset-interop-core`).

use std::task::{Context, Poll};

use sciencekit_common::SKError;

use crate::batch::SKTabularBatch;

/// An asynchronous source of tabular batches.
///
/// This is the minimal poll-based contract an async engine (DataFusion,
/// Polars `collect_batches`, an Arrow flight stream) implements to feed the
/// synchronous streaming driver. It deliberately avoids a `futures`
/// dependency: the bridge polls it on a dedicated thread, so the caller's
/// asynchronous runtime is never blocked.
pub trait SKAsyncTabularStream: Send {
    /// Poll the next batch, returning `Pending` until one is ready.
    fn poll_next_batch(
        &mut self,
        context: &mut Context<'_>,
    ) -> Poll<Option<Result<SKTabularBatch, SKError>>>;
}
