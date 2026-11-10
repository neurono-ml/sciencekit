//! The reopenable, fallible tabular source contract (spec
//! `dataset-interop-core`).

use sciencekit_common::SKError;

use crate::batch::SKTabularBatch;

/// A fallible supplier of tabular batches that can be opened per pass.
///
/// An adapter implements this trait over its engine; the provider bridge then
/// exposes it as the library's feature-only and supervised streaming providers.
/// A reopenable source returns to the beginning on every [`open`](Self::open),
/// so an iterative estimator receives the same batches in the same order on
/// every pass. A one-shot source accepts a single open.
pub trait SKTabularSource: Send {
    /// Whether the source can be reopened for another pass.
    const IS_REOPENABLE: bool = false;

    /// Open a pass, repositioning a reopenable source at its beginning.
    ///
    /// # Errors
    /// Reports a reopen or engine failure through the central taxonomy.
    fn open(&mut self) -> Result<(), SKError>;

    /// Produce the next batch, `None` at the end of the pass.
    ///
    /// # Errors
    /// Reports a read failure through the central taxonomy; no partial batch is
    /// delivered.
    fn next_batch(&mut self) -> Result<Option<SKTabularBatch>, SKError>;
}
