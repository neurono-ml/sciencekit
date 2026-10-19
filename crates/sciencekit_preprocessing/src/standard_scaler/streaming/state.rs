//! The combinable Welford state and its Chan et al. merge.

use ndarray::{Array1, azip};
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

/// Chan et al. merge of one batch into running moments: exact up to float
/// rounding, hence independent of batch order and sizes. Shared by the driver
/// state and the model's incremental `partial_fit`.
pub(crate) fn merge_moments<F: SKFloat>(
    count: &mut u64,
    mean: &mut Array1<F>,
    deviations: &mut Array1<F>,
    batch_mean: &Array1<F>,
    batch_deviations: &Array1<F>,
    batch_count: u64,
) {
    if *count == 0 {
        mean.assign(batch_mean);
        deviations.assign(batch_deviations);
        *count = batch_count;
        return;
    }
    let previous_count_as_scalar = F::from(*count).expect("row count fits in scalar");
    let batch_count_as_scalar = F::from(batch_count).expect("row count fits in scalar");
    let total_as_scalar = F::from(*count + batch_count).expect("row count fits in scalar");
    azip!(
        (mean in mean.view_mut(),
         deviations in deviations.view_mut(),
         batch_mean in batch_mean.view(),
         batch_deviations in batch_deviations.view())
        {
            let delta = *batch_mean - *mean;
            *mean = *mean + delta * batch_count_as_scalar / total_as_scalar;
            *deviations = *deviations
                + *batch_deviations
                + delta * delta * previous_count_as_scalar * batch_count_as_scalar / total_as_scalar;
        }
    );
    *count += batch_count;
}

/// Chan et al. merge of one batch into the running state: exact up to float
/// rounding, hence independent of batch order and sizes.
pub(super) fn merge_batch<F: SKFloat>(
    state: &mut StreamingMoments<F>,
    batch_mean: &Array1<F>,
    batch_deviations: &Array1<F>,
    batch_count: u64,
) {
    merge_moments(
        &mut state.count,
        &mut state.mean,
        &mut state.sum_squared_deviations,
        batch_mean,
        batch_deviations,
        batch_count,
    );
}
