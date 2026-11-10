//! Hand-computed tests for the regression impurities (spec `decision-trees`).

use super::{sk_absolute_error_impurity, sk_squared_error_impurity};

fn assert_close(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() < 1e-12,
        "expected {expected}, found {actual}"
    );
}

/// A constant node has zero squared-error and absolute-error impurity.
#[test]
fn constant_node_has_zero_impurity() {
    assert_eq!(sk_squared_error_impurity(&[4.0, 4.0, 4.0]), 0.0);
    assert_eq!(sk_absolute_error_impurity(&[4.0, 4.0, 4.0]), 0.0);
}

/// Responses `1, 2, 3` have variance `2/3` and median absolute deviation
/// `2/3`.
#[test]
fn three_points_match_the_hand_computation() {
    assert_close(sk_squared_error_impurity(&[1.0, 2.0, 3.0]), 2.0 / 3.0);
    assert_close(sk_absolute_error_impurity(&[1.0, 2.0, 3.0]), 2.0 / 3.0);
}

/// Responses `1, 2, 6` have variance `14/3` and absolute deviation `5/3`.
#[test]
fn skewed_responses_match_the_hand_computation() {
    assert_close(sk_squared_error_impurity(&[1.0, 2.0, 6.0]), 14.0 / 3.0);
    assert_close(sk_absolute_error_impurity(&[1.0, 2.0, 6.0]), 5.0 / 3.0);
}

/// An even count averages the two middle responses before measuring the
/// absolute deviation.
#[test]
fn even_count_uses_the_averaged_median() {
    assert_close(sk_squared_error_impurity(&[1.0, 2.0, 3.0, 4.0]), 1.25);
    assert_close(sk_absolute_error_impurity(&[1.0, 2.0, 3.0, 4.0]), 1.0);
}

/// Empty vectors are treated as pure.
#[test]
fn empty_responses_are_pure() {
    let empty: [f64; 0] = [];
    assert_eq!(sk_squared_error_impurity(&empty), 0.0);
    assert_eq!(sk_absolute_error_impurity(&empty), 0.0);
}

/// The impurities are generic over both supported floats.
#[test]
fn regression_impurity_accepts_either_float() {
    assert_eq!(sk_squared_error_impurity(&[1.0_f32, 3.0]), 1.0_f32);
    assert_eq!(sk_squared_error_impurity(&[1.0_f64, 3.0]), 1.0_f64);
}
