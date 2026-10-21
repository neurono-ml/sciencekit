//! The sequential, fallible supervised streaming source (spec
//! `streaming-batches`).

use crate::SKError;

use super::supervised_batch::SKSupervisedBatch;

/// A sequential supervised streaming source exposed as fallible block
/// iteration.
///
/// The supervised companion of [`SKLazySource`](super::SKLazySource): each block
/// pairs a feature matrix with its targets, so a supervised algorithm can fit
/// out-of-core without materializing the whole dataset. Intermediate read
/// failures yield an error from the central taxonomy instead of panicking.
pub trait SKSupervisedLazySource<F> {
    /// The iteration error type; converts from the central [`SKError`].
    type Error: From<SKError>;
    /// Produce the next supervised batch, `None` at the end of a finite source.
    fn next_supervised_batch(&mut self) -> Result<Option<SKSupervisedBatch<F>>, Self::Error>;
}
