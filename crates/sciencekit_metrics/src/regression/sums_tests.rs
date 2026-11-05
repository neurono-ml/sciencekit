//! TDD tests for the Neumaier compensated sum (spec `evaluation-metrics`).

use super::SKCompensatedSum;

/// The compensated sum keeps small addends next to large values.
#[test]
fn compensated_sum_keeps_small_addends_next_to_large_values() {
    let mut compensated: SKCompensatedSum<f32> = SKCompensatedSum::new();
    for value in [1.0e8_f32, 1.0, -1.0e8, 1.0] {
        compensated.add(value);
    }
    assert!((compensated.value() - 2.0).abs() < 1e-6);

    let naive: f32 = [1.0e8_f32, 1.0, -1.0e8, 1.0].iter().sum();
    assert!(naive.abs() < 1.5, "the naive sum loses one unit: {naive}");
}

/// Merging two partial sums equals adding every value sequentially.
#[test]
fn merging_partial_sums_equals_one_sequential_pass() {
    let values = [1.0e6_f64, 1.0, 2.0, -1.0e6, 3.0, 4.0];
    let mut sequential = SKCompensatedSum::new();
    for value in values {
        sequential.add(value);
    }
    let mut left = SKCompensatedSum::new();
    let mut right = SKCompensatedSum::new();
    for value in &values[..3] {
        left.add(*value);
    }
    for value in &values[3..] {
        right.add(*value);
    }
    left.merge(right);
    assert!((left.value() - sequential.value()).abs() < 1e-12);
    assert!((left.value() - 10.0).abs() < 1e-12);
}
