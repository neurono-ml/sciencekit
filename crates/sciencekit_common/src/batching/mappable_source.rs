//! The abstract random-access streaming source.

use ndarray::ArrayView1;

use crate::SKError;

/// An abstract random-access source: direct positional access to data units,
/// independent of the storage mechanism (memmap arrives at interop).
pub trait SKMappableSource<F> {
    /// The access error type; converts from the central [`SKError`].
    type Error: From<SKError>;
    /// The number of data rows (units).
    fn number_of_rows(&self) -> usize;
    /// Access a row by index without traversing previous rows.
    fn row(&self, index: usize) -> Result<ArrayView1<'_, F>, Self::Error>;
}
