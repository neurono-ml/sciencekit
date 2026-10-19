//! The shared robust-scaler configuration and its masking / unit-variance rule.

use ndarray::Array1;
use sciencekit_common::SKFloat;

use crate::robust_scaler::quantiles::standard_normal_interquantile_span;

/// The immutable per-scaler options shared by the estimator and its model:
/// the two masking flags, the quantile range and the unit-variance flag.
#[derive(Debug, Clone, Copy)]
pub(crate) struct SKRobustScalerConfiguration<F> {
    pub(super) with_centering: bool,
    pub(super) with_scaling: bool,
    pub(super) quantile_low: F,
    pub(super) quantile_high: F,
    pub(super) unit_variance: bool,
}

impl<F: SKFloat> SKRobustScalerConfiguration<F> {
    /// The `(low, high)` percentile pair the scale is built from.
    pub(super) fn quantile_range(&self) -> (F, F) {
        (self.quantile_low, self.quantile_high)
    }

    /// The standard-normal interquantile span used as the unit-variance
    /// divisor.
    pub(super) fn unit_variance_span(&self) -> F {
        F::from(standard_normal_interquantile_span(
            self.quantile_low.to_f64().unwrap_or(25.0),
            self.quantile_high.to_f64().unwrap_or(75.0),
        ))
        .unwrap_or(F::one())
    }
}

/// Apply the masking flags and the unit-variance divisor to the raw quantile
/// statistics, keeping every scale positive so the output stays finite. Shared
/// by the dense fit and the online update.
pub(crate) fn finalize_statistics<F: SKFloat>(
    configuration: &SKRobustScalerConfiguration<F>,
    raw_center: &Array1<F>,
    raw_scale: &Array1<F>,
) -> (Array1<F>, Array1<F>) {
    let raw_span = if configuration.unit_variance {
        configuration.unit_variance_span()
    } else {
        F::one()
    };
    // A degenerate quantile range yields a zero (or non-finite) span; fall back
    // to a unit divisor so the scale stays finite.
    let span = if raw_span > F::zero() && raw_span.is_finite() {
        raw_span
    } else {
        F::one()
    };
    let center: Array1<F> = raw_center
        .iter()
        .map(|&value| {
            if configuration.with_centering {
                value
            } else {
                F::zero()
            }
        })
        .collect();
    let scale: Array1<F> = raw_scale
        .iter()
        .map(|&value| {
            let scaled = if configuration.unit_variance {
                value / span
            } else {
                value
            };
            if configuration.with_scaling && scaled > F::zero() && scaled.is_finite() {
                scaled
            } else {
                F::one()
            }
        })
        .collect();
    (center, scale)
}
