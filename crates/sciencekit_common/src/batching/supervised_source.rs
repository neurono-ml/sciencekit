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
    /// The iteration error type; converts to and from the central [`SKError`].
    type Error: From<SKError> + Into<SKError>;
    /// Produce the next supervised batch, `None` at the end of a finite source.
    fn next_supervised_batch(&mut self) -> Result<Option<SKSupervisedBatch<F>>, Self::Error>;
}

/// A mutable reference to a supervised source is itself a source, forwarding
/// every read.
///
/// This keeps the one-pass provider's reborrow ([`&mut S`]) a valid
/// [`SKSupervisedLazySource`] without wrapping the concrete source.
impl<F, Source> SKSupervisedLazySource<F> for &mut Source
where
    Source: SKSupervisedLazySource<F>,
{
    type Error = Source::Error;

    fn next_supervised_batch(&mut self) -> Result<Option<SKSupervisedBatch<F>>, Self::Error> {
        (**self).next_supervised_batch()
    }
}
