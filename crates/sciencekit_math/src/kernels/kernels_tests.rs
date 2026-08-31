//! TDD tests for the higher-order kernels (spec `higher-order-kernels`).

use ndarray::{Array2, array};
use sciencekit_common::SKFloat;

use super::{sk_axis_sum, sk_binary_combine, sk_elementwise_transform, sk_scale_in_place};

/// Elementwise `x → 2x + 1` produces `2 * input + 1` for every element.
#[test]
fn elementwise_transform_applies_mapping() {
    let input = array![1.0_f64, 2.0, 3.0, 4.0];
    let output = sk_elementwise_transform(&input.view(), |x| 2.0 * x + 1.0);
    assert_eq!(output, array![3.0, 5.0, 7.0, 9.0]);
}

/// Elementwise transform works identically on a single element (small data).
#[test]
fn elementwise_transform_single_element() {
    let input = array![7.0_f64];
    let output = sk_elementwise_transform(&input.view(), |x| x * x);
    assert_eq!(output, array![49.0]);
}

/// Binary combine pairs corresponding elements and yields a new owned array.
#[test]
fn binary_combine_pairs_corresponding_elements() {
    let left = array![1.0_f64, 2.0, 3.0];
    let right = array![10.0_f64, 20.0, 30.0];
    let output = sk_binary_combine(&left.view(), &right.view(), |l, r| l + r);
    assert_eq!(output, array![11.0, 22.0, 33.0]);
}

/// Binary combine is generic over the scalar type.
#[test]
fn binary_combine_works_on_f32() {
    let left = array![1.0_f32, 2.0];
    let right = array![0.5_f32, 0.25];
    let output = sk_binary_combine(&left.view(), &right.view(), |l, r| l * r);
    assert_eq!(output, array![0.5, 0.5]);
}

/// Column sum (axis 0) matches the per-column totals on a row-major array.
#[test]
fn axis_zero_sum_matches_column_totals() {
    let input: Array2<f64> = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];
    let sums = sk_axis_sum(&input.view(), 0);
    assert_eq!(sums, array![12.0, 15.0, 18.0]);
}

/// Row sum (axis 1) matches the per-row totals on a row-major array.
#[test]
fn axis_one_sum_matches_row_totals() {
    let input: Array2<f64> = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];
    let sums = sk_axis_sum(&input.view(), 1);
    assert_eq!(sums, array![6.0, 15.0, 24.0]);
}

/// In-place scaling mutates the buffer and returns nothing.
#[test]
fn in_place_scale_mutates_without_copy() {
    let mut input: Array2<f64> = array![[1.0, 2.0], [3.0, 4.0]];
    sk_scale_in_place(&mut input, 3.0);
    // No new array is returned; the input buffer is modified in place.
    assert_eq!(input, array![[3.0, 6.0], [9.0, 12.0]]);
}

/// The kernels stay generic over both supported floats.
#[test]
fn kernels_accept_either_float() {
    fn accepts<F: SKFloat>(input: &Array2<F>) {
        let _ = sk_axis_sum(&input.view(), 0);
    }
    accepts(&array![[1.0_f32, 2.0]]);
    accepts(&array![[1.0_f64, 2.0]]);
}
