//! Shared candidate bookkeeping and batch-grain decisions for both trees.

use sciencekit_common::SKFloat;

/// The minimum query-rows × training-rows product that enables the row-parallel
/// query path.
pub(crate) const SK_QUERY_PARALLEL_GRAIN: usize = 1 << 15;

/// One scored training row during a search.
#[derive(Debug, Clone, Copy)]
pub(crate) struct SKCandidate<F> {
    pub(crate) distance: F,
    pub(crate) index: usize,
}

/// Whether a batch crosses the parallel grain.
pub(crate) fn sk_should_run_parallel(query_rows: usize, training_rows: usize) -> bool {
    query_rows.saturating_mul(training_rows) >= SK_QUERY_PARALLEL_GRAIN
}

/// The current worst kept distance, or infinity while fewer than `k`
/// candidates are held.
pub(crate) fn sk_current_worst<F: SKFloat>(best: &[SKCandidate<F>], neighbors_count: usize) -> F {
    if best.len() < neighbors_count {
        F::infinity()
    } else {
        best[neighbors_count - 1].distance
    }
}

/// Insert a candidate into the bounded best list, keeping it sorted by
/// `(distance, training_index)` and at most `neighbors_count` entries.
pub(crate) fn sk_offer_candidate<F: SKFloat>(
    best: &mut Vec<SKCandidate<F>>,
    distance: F,
    index: usize,
    neighbors_count: usize,
) {
    if best.len() == neighbors_count {
        let worst = best[neighbors_count - 1];
        let better =
            distance < worst.distance || (distance == worst.distance && index < worst.index);
        if !better {
            return;
        }
    }
    let position = best.partition_point(|candidate| {
        candidate.distance < distance || (candidate.distance == distance && candidate.index < index)
    });
    best.insert(position, SKCandidate { distance, index });
    if best.len() > neighbors_count {
        best.pop();
    }
}

/// Sort candidates by distance ascending and ties by the smallest training row
/// index.
pub(crate) fn sk_sort_candidates<F: SKFloat>(candidates: &mut [SKCandidate<F>]) {
    candidates.sort_by(|left, right| {
        left.distance
            .partial_cmp(&right.distance)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| left.index.cmp(&right.index))
    });
}
