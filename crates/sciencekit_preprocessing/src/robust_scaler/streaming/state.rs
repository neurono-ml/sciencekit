//! The mutable streaming sketch state.

use sciencekit_common::SKError;
use tdigest::TDigest;

/// Combinable per-column online quantile sketch state: the row count, the
/// column count and one sketch per column. Feeding a batch extends every column
/// sketch, so the state stays bounded by `columns × compression`.
pub(super) struct StreamingSketches {
    pub(super) count: u64,
    pub(super) columns_count: usize,
    pub(super) sketches: Vec<TDigest>,
    pub(super) error: Option<SKError>,
}

impl StreamingSketches {
    pub(super) fn empty() -> Self {
        Self {
            count: 0,
            columns_count: 0,
            sketches: Vec::new(),
            error: None,
        }
    }
}
