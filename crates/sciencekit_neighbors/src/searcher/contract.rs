//! The shared exact-search contract (spec `nearest-neighbor-search`).

use sciencekit_common::{SKDataView, SKError};

use super::{SKNeighborQuery, SKRadiusNeighbors};

/// The exact nearest-neighbor searcher contract shared by both trees and, in a
/// later change, by the brute-force searcher.
///
/// The contract is object-safe: every method is generic only over the trait's
/// scalar parameter, so an estimator can keep a runtime `Box<dyn
/// SKNearestNeighborsSearcher<F>>` without leaking the concrete index type into
/// its fitted model.
pub trait SKNearestNeighborsSearcher<F> {
    /// Return the exact `k` nearest training rows for every query row.
    ///
    /// Distances are ascending and ties are broken by the smallest training row
    /// index. The result has shape `(query_rows, neighbors_count)`.
    fn k_nearest(
        &self,
        queries: SKDataView<'_, F>,
        neighbors_count: usize,
    ) -> Result<SKNeighborQuery<F>, SKError>;

    /// Return every training row within the inclusive `radius` for every query
    /// row, with its distance.
    fn radius_neighbors(
        &self,
        queries: SKDataView<'_, F>,
        radius: F,
    ) -> Result<SKRadiusNeighbors<F>, SKError>;
}
