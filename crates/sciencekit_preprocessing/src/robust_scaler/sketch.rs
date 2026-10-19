//! Per-column online quantile sketches for the robust scaler.
//!
//! Dense fitting stays exact (exact order statistics); the incremental and
//! out-of-core paths maintain one `tdigest` sketch per column so the center and
//! scale can be learned from a stream without buffering the whole column. The
//! sketch is fed per batch, so memory is bounded by
//! `columns × compression`, independent of the row count — the out-of-core
//! property scikit-learn's `RobustScaler` cannot offer.

use ndarray::{Array1, ArrayView2};
use sciencekit_common::SKFloat;
use tdigest::TDigest;

use crate::robust_scaler::core_implementation::SKRobustScalerConfiguration;

/// Internal compression parameter for the per-column sketch: a
/// memory/accuracy trade-off, deliberately not a builder option so the public
/// surface stays focused (spec `robust-scaler`).
const TDIGEST_COMPRESSION: usize = 200;

/// A fresh, empty sketch per column.
pub(crate) fn new_sketches(columns_count: usize) -> Vec<TDigest> {
    (0..columns_count)
        .map(|_| TDigest::new_with_size(TDIGEST_COMPRESSION))
        .collect()
}

/// Feed every row of one batch into the per-column sketches and compress them,
/// so quantile queries see the new values.
pub(crate) fn extend_sketches<F: SKFloat>(sketches: &mut [TDigest], features: &ArrayView2<F>) {
    for (column, sketch) in sketches.iter_mut().enumerate() {
        sketch.extend_values(
            features
                .column(column)
                .iter()
                .map(|&value| value.to_f64().unwrap_or(0.0)),
        );
        sketch.flush();
    }
}

/// The raw per-column `(median, interquantile range)` read from the sketches.
pub(crate) fn sketch_statistics<F: SKFloat>(
    sketches: &[TDigest],
    configuration: &SKRobustScalerConfiguration<F>,
) -> (Array1<F>, Array1<F>) {
    let (low, high) = configuration.quantile_range();
    let low = low.to_f64().unwrap_or(25.0);
    let high = high.to_f64().unwrap_or(75.0);
    let mut centers = Array1::zeros(sketches.len());
    let mut scales = Array1::zeros(sketches.len());
    for (index, sketch) in sketches.iter().enumerate() {
        centers[index] = quantile_as_scalar(sketch, 50.0);
        let lower = quantile_value(sketch, low);
        let upper = quantile_value(sketch, high);
        scales[index] = F::from(upper - lower).unwrap_or(F::zero());
    }
    (centers, scales)
}

/// One quantile of a sketch, converted back to the working scalar type.
fn quantile_as_scalar<F: SKFloat>(sketch: &TDigest, percent: f64) -> F {
    F::from(quantile_value(sketch, percent)).unwrap_or(F::zero())
}

/// One quantile (given in percent) of a sketch, as `f64`.
fn quantile_value(sketch: &TDigest, percent: f64) -> f64 {
    sketch.estimate_quantile(percent / 100.0).unwrap_or(0.0)
}
