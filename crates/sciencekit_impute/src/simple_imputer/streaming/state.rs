//! The state threaded through the streaming driver (spec `simple-imputer`).

use sciencekit_common::SKError;

use crate::simple_imputer::state::SKImputationState;

/// The accumulated state of one out-of-core fit: the per-column statistics
/// state, the discovered column count and the row count.
pub(in crate::simple_imputer) struct StreamingImputation<F> {
    pub(super) columns_count: usize,
    pub(super) rows_count: u64,
    pub(super) imputation: Option<SKImputationState<F>>,
    pub(super) error: Option<SKError>,
}

impl<F> StreamingImputation<F> {
    /// A state waiting for the first batch to reveal the column count.
    pub(super) fn empty() -> Self {
        Self {
            columns_count: 0,
            rows_count: 0,
            imputation: None,
            error: None,
        }
    }
}
