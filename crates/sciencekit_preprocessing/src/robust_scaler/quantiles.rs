//! Exact dense quantiles and the standard-normal interquantile span.
//!
//! Dense fitting uses exact order-statistic quantiles (the scikit-learn / NumPy
//! linear-interpolation convention) so it matches scikit-learn bit for bit. The
//! standard-normal span for `unit_variance` is a self-contained rational
//! approximation of the probit function (Acklam), avoiding a statistics
//! dependency for one constant per quantile range.

use ndarray::{Array1, ArrayView2};
use sciencekit_common::SKFloat;

/// The exact per-column `(median, interquantile range)` over the percentile
/// pair `[quantile_low, quantile_high]`, computed from the sorted column.
pub(super) fn column_quantile_statistics<F: SKFloat>(
    features: &ArrayView2<F>,
    quantile_low: F,
    quantile_high: F,
) -> (Array1<F>, Array1<F>) {
    let (_, columns_count) = features.dim();
    let hundred = F::from(100.0).expect("100 fits in scalar");
    let fifty = F::from(50.0).expect("50 fits in scalar");
    let mut centers = Array1::zeros(columns_count);
    let mut scales = Array1::zeros(columns_count);
    for column in 0..columns_count {
        let mut values: Vec<F> = features.column(column).iter().copied().collect();
        values.sort_by(|left, right| left.partial_cmp(right).unwrap_or(std::cmp::Ordering::Equal));
        centers[column] = exact_quantile(&values, fifty / hundred);
        let low = exact_quantile(&values, quantile_low / hundred);
        let high = exact_quantile(&values, quantile_high / hundred);
        scales[column] = high - low;
    }
    (centers, scales)
}

/// The exact quantile of an ascending slice at `probability` in `[0, 1]`, using
/// linear interpolation between the two bracketing order statistics.
pub(super) fn exact_quantile<F: SKFloat>(sorted_values: &[F], probability: F) -> F {
    let count = sorted_values.len();
    if count == 0 {
        return F::zero();
    }
    if count == 1 {
        return sorted_values[0];
    }
    let position = probability * F::from(count - 1).expect("index fits in scalar");
    let lower = position.floor().to_usize().unwrap_or(0);
    let upper = position.ceil().to_usize().unwrap_or(lower);
    if lower == upper {
        sorted_values[lower]
    } else {
        let fraction = position - F::from(lower).expect("index fits in scalar");
        sorted_values[lower] + (sorted_values[upper] - sorted_values[lower]) * fraction
    }
}

/// The standard-normal interquantile span for the percentile pair
/// `[low_percent, high_percent]`, i.e. `Φ⁻¹(high) − Φ⁻¹(low)`.
pub(super) fn standard_normal_interquantile_span(low_percent: f64, high_percent: f64) -> f64 {
    probit(high_percent / 100.0) - probit(low_percent / 100.0)
}

/// Acklam's rational approximation of the inverse standard-normal CDF.
///
/// Accurate to roughly `1.15e-9` over the open interval; the exact `±∞` limits
/// are returned at the closed ends.
fn probit(probability: f64) -> f64 {
    const A: [f64; 6] = [
        -3.969_683_028_665_376e1,
        2.209_460_984_245_205e2,
        -2.759_285_104_469_687e2,
        1.383_577_518_672_69e2,
        -3.066_479_806_614_716e1,
        2.506_628_277_459_239,
    ];
    const B: [f64; 5] = [
        -5.447_609_879_822_406e1,
        1.615_858_368_580_409e2,
        -1.556_989_798_598_866e2,
        6.680_131_188_771_972e1,
        -1.328_068_155_288_572e1,
    ];
    const C: [f64; 6] = [
        -7.784_894_002_430_293e-3,
        -3.223_964_580_411_365e-1,
        -2.400_758_277_161_838,
        -2.549_732_539_343_734,
        4.374_664_141_464_968,
        2.938_163_982_698_783,
    ];
    const D: [f64; 4] = [
        7.784_695_709_041_462e-3,
        3.224_671_290_700_398e-1,
        2.445_134_137_142_996,
        3.754_408_661_907_416,
    ];
    const LOW: f64 = 0.02425;
    const HIGH: f64 = 1.0 - LOW;

    if probability <= 0.0 {
        return f64::NEG_INFINITY;
    }
    if probability >= 1.0 {
        return f64::INFINITY;
    }
    if probability < LOW {
        let q = (-2.0 * probability.ln()).sqrt();
        (((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    } else if probability <= HIGH {
        let q = probability - 0.5;
        let r = q * q;
        (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q
            / (((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0)
    } else {
        let q = (-2.0 * (1.0 - probability).ln()).sqrt();
        -(((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    }
}
