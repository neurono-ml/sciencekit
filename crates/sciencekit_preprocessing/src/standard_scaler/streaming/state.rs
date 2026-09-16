//! The combinable Welford state and its Chan et al. merge.

use ndarray::Array1;
use sciencekit_common::SKFloat;

/// Combinable per-column Welford state: count, mean and sum of squared
/// deviations. Merging two states is exact (up to float rounding), so batch
/// order never matters.
pub(super) struct StreamingMoments<F> {
    pub(super) count: u64,
    pub(super) columns_count: usize,
    pub(super) mean: Array1<F>,
    pub(super) sum_squared_deviations: Array1<F>,
    pub(super) error: Option<sciencekit_common::SKError>,
    pub(super) available_memory_bytes: u64,
    pub(super) cpu_cores: usize,
}

impl<F: SKFloat> StreamingMoments<F> {
    pub(super) fn empty(available_memory_bytes: u64, cpu_cores: usize) -> Self {
        Self {
            count: 0,
            columns_count: 0,
            mean: Array1::zeros(0),
            sum_squared_deviations: Array1::zeros(0),
            error: None,
            available_memory_bytes,
            cpu_cores,
        }
    }
}

/// Chan et al. merge of one batch into the running state: exact up to float
/// rounding, hence independent of batch order and sizes.
pub(super) fn merge_batch<F: SKFloat>(
    state: &mut StreamingMoments<F>,
    batch_mean: &Array1<F>,
    batch_deviations: &Array1<F>,
    batch_count: u64,
) {
    if state.count == 0 {
        state.mean.assign(batch_mean);
        state.sum_squared_deviations.assign(batch_deviations);
        state.count = batch_count;
        return;
    }
    let previous_count_as_scalar = F::from(state.count).expect("row count fits in scalar");
    let batch_count_as_scalar = F::from(batch_count).expect("row count fits in scalar");
    let total_as_scalar = F::from(state.count + batch_count).expect("row count fits in scalar");
    for index in 0..state.columns_count {
        let delta = batch_mean[index] - state.mean[index];
        state.mean[index] = state.mean[index] + delta * batch_count_as_scalar / total_as_scalar;
        state.sum_squared_deviations[index] = state.sum_squared_deviations[index]
            + batch_deviations[index]
            + delta * delta * previous_count_as_scalar * batch_count_as_scalar / total_as_scalar;
    }
    state.count += batch_count;
}
