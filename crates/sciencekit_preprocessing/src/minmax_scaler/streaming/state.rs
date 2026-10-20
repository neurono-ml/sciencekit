//! The combinable running-extrema state and its element-wise merge.

use ndarray::{Array1, azip};
use sciencekit_common::SKFloat;

/// Combinable per-column extrema state: count, minimum and range
/// (`maximum - minimum`). Merging two states is exact (min/max are
/// order-independent), so batch order never matters.
pub(super) struct StreamingExtrema<F> {
    pub(super) count: u64,
    pub(super) columns_count: usize,
    pub(super) minimum: Array1<F>,
    pub(super) maximum: Array1<F>,
    pub(super) data_range: Array1<F>,
    pub(super) error: Option<sciencekit_common::SKError>,
    pub(super) available_memory_bytes: u64,
    pub(super) cpu_cores: usize,
}

impl<F: SKFloat> StreamingExtrema<F> {
    pub(super) fn empty(available_memory_bytes: u64, cpu_cores: usize) -> Self {
        Self {
            count: 0,
            columns_count: 0,
            minimum: Array1::zeros(0),
            maximum: Array1::zeros(0),
            data_range: Array1::zeros(0),
            error: None,
            available_memory_bytes,
            cpu_cores,
        }
    }
}

/// Merge one batch's extrema into the running minimum, maximum and range. The
/// extrema are stored directly — never reconstructed — so an incremental update
/// is bit-identical to the dense fit. Shared by the driver state and the model's
/// incremental `partial_fit`.
pub(crate) fn merge_extrema<F: SKFloat>(
    minimum: &mut Array1<F>,
    maximum: &mut Array1<F>,
    data_range: &mut Array1<F>,
    batch_minimum: &Array1<F>,
    batch_maximum: &Array1<F>,
) {
    azip!(
        (minimum in minimum.view_mut(),
         maximum in maximum.view_mut(),
         data_range in data_range.view_mut(),
         batch_minimum in batch_minimum.view(),
         batch_maximum in batch_maximum.view())
        {
            let new_minimum = minimum.min(*batch_minimum);
            let new_maximum = maximum.max(*batch_maximum);
            *minimum = new_minimum;
            *maximum = new_maximum;
            *data_range = new_maximum - new_minimum;
        }
    );
}

/// Merge one batch into the running streaming state.
pub(super) fn merge_batch<F: SKFloat>(
    state: &mut StreamingExtrema<F>,
    batch_minimum: &Array1<F>,
    batch_maximum: &Array1<F>,
    batch_count: u64,
) {
    merge_extrema(
        &mut state.minimum,
        &mut state.maximum,
        &mut state.data_range,
        batch_minimum,
        batch_maximum,
    );
    state.count += batch_count;
}
