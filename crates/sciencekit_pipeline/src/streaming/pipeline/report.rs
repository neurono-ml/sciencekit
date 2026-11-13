//! Metrics describing one streaming fit.

/// Metrics describing one streaming fit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SKStreamingFitReport {
    pub(crate) number_of_stages: usize,
    pub(crate) number_of_epochs: usize,
    pub(crate) number_of_batches: usize,
    pub(crate) number_of_rows: usize,
    pub(crate) materialized_bytes: usize,
    pub(crate) finalizer_bytes: usize,
}

impl SKStreamingFitReport {
    /// The number of transformer stages fitted.
    pub fn number_of_stages(&self) -> usize {
        self.number_of_stages
    }

    /// The configured epoch count (one for a one-shot provider).
    pub fn number_of_epochs(&self) -> usize {
        self.number_of_epochs
    }

    /// The number of batches read in the finalizer pass.
    pub fn number_of_batches(&self) -> usize {
        self.number_of_batches
    }

    /// The number of rows read in the finalizer pass.
    pub fn number_of_rows(&self) -> usize {
        self.number_of_rows
    }

    /// The resident bytes of a materialized one-shot source, zero when the
    /// provider was reopened directly.
    pub fn materialized_bytes(&self) -> usize {
        self.materialized_bytes
    }

    /// The resident bytes the finalizer accumulated.
    pub fn finalizer_bytes(&self) -> usize {
        self.finalizer_bytes
    }
}
