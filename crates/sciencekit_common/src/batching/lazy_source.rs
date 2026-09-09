//! The sequential, fallible streaming source.

use crate::SKError;

use super::data_batch::SKDataBatch;

/// A sequential streaming source exposed as fallible block iteration.
///
/// Intermediate read failures yield an error from the central taxonomy instead
/// of panicking; the consumer decides whether to stop or handle it.
pub trait SKLazySource<F> {
    /// The iteration error type; converts from the central [`SKError`].
    type Error: From<SKError>;
    /// Produce the next batch, `None` at the end of a finite source.
    fn next_batch(&mut self) -> Result<Option<SKDataBatch<F>>, Self::Error>;
}
