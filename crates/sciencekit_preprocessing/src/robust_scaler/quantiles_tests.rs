//! Companion tests for the exact-quantile helper and the standard-normal span.

use super::quantiles::{exact_quantile, standard_normal_interquantile_span};

/// Exact quantiles on the hand-checkable column `[1, 2, 3, 4, 5]`.
#[test]
fn exact_quantile_matches_hand_computed_values() {
    let values = [1.0, 2.0, 3.0, 4.0, 5.0];
    assert!((exact_quantile(&values, 0.0) - 1.0).abs() < 1e-12);
    assert!((exact_quantile(&values, 0.25) - 2.0).abs() < 1e-12);
    assert!((exact_quantile(&values, 0.5) - 3.0).abs() < 1e-12);
    assert!((exact_quantile(&values, 0.75) - 4.0).abs() < 1e-12);
    assert!((exact_quantile(&values, 1.0) - 5.0).abs() < 1e-12);
}

/// Between two order statistics the helper interpolates linearly.
#[test]
fn exact_quantile_interpolates_between_order_statistics() {
    let values = [1.0, 2.0, 3.0, 4.0];
    // p = 0.1 → position 0.3 → 1 + 0.3 × (2 − 1) = 1.3
    assert!((exact_quantile(&values, 0.1) - 1.3).abs() < 1e-12);
}

/// A single value is its own quantile; an empty slice yields zero.
#[test]
fn exact_quantile_single_and_empty() {
    assert_eq!(exact_quantile(&[7.0], 0.5), 7.0);
    assert_eq!(exact_quantile(&[], 0.5), 0.0);
}

/// The standard-normal span for `(25, 75)` is the well-known ≈ 1.34898.
#[test]
fn standard_normal_span_matches_known_value() {
    let span = standard_normal_interquantile_span(25.0, 75.0);
    assert!(
        (span - 1.348_979_500_392_163_4).abs() < 1e-9,
        "span = {span}"
    );
}

/// The span is symmetric about the median and grows with the range width.
#[test]
fn standard_normal_span_is_symmetric_and_widens() {
    let narrow = standard_normal_interquantile_span(40.0, 60.0);
    let wide = standard_normal_interquantile_span(10.0, 90.0);
    assert!(wide > narrow);
    let lower = standard_normal_interquantile_span(10.0, 50.0);
    let upper = standard_normal_interquantile_span(50.0, 90.0);
    assert!((lower - upper).abs() < 1e-9);
}
