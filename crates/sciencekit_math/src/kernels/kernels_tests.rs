//! TDD tests for the higher-order kernels (spec `higher-order-kernels`).

use ndarray::{Array, Array1, Array2, array};
use sciencekit_common::SKFloat;

use super::{sk_axis_sum, sk_binary_combine, sk_elementwise_transform, sk_scale_in_place};

/// Assert two float vectors agree within a relative floating-point tolerance
/// (the repo's `assert_close` contract — parallel reordering may shift the last
/// bits of reductions).
fn assert_close(a: &Array<f64, ndarray::Ix1>, b: &Array<f64, ndarray::Ix1>) {
    assert_eq!(a.len(), b.len());
    for (x, y) in a.iter().zip(b.iter()) {
        let tolerance = 1e-9 * (1.0 + x.abs().max(y.abs()));
        assert!(
            (x - y).abs() <= tolerance,
            "values {x} and {y} differ beyond tolerance {tolerance}"
        );
    }
}

/// Elementwise `x → 2x + 1` produces `2 * input + 1` for every element.
#[test]
fn elementwise_transform_applies_mapping() {
    let input = array![1.0_f64, 2.0, 3.0, 4.0];
    let output = sk_elementwise_transform(&input.view(), |x| 2.0 * x + 1.0, 1);
    assert_eq!(output, array![3.0, 5.0, 7.0, 9.0]);
}

/// Elementwise transform works identically on a single element (small data).
#[test]
fn elementwise_transform_single_element() {
    let input = array![7.0_f64];
    let output = sk_elementwise_transform(&input.view(), |x| x * x, 1);
    assert_eq!(output, array![49.0]);
}

/// Binary combine pairs corresponding elements and yields a new owned array.
#[test]
fn binary_combine_pairs_corresponding_elements() {
    let left = array![1.0_f64, 2.0, 3.0];
    let right = array![10.0_f64, 20.0, 30.0];
    let output = sk_binary_combine(&left.view(), &right.view(), |l, r| l + r, 1);
    assert_eq!(output, array![11.0, 22.0, 33.0]);
}

/// Binary combine is generic over the scalar type.
#[test]
fn binary_combine_works_on_f32() {
    let left = array![1.0_f32, 2.0];
    let right = array![0.5_f32, 0.25];
    let output = sk_binary_combine(&left.view(), &right.view(), |l, r| l * r, 1);
    assert_eq!(output, array![0.5, 0.5]);
}

/// Column sum (axis 0) matches the per-column totals on a row-major array.
#[test]
fn axis_zero_sum_matches_column_totals() {
    let input: Array2<f64> = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];
    let sums = sk_axis_sum(&input.view(), 0, 1);
    assert_eq!(sums, array![12.0, 15.0, 18.0]);
}

/// Row sum (axis 1) matches the per-row totals on a row-major array.
#[test]
fn axis_one_sum_matches_row_totals() {
    let input: Array2<f64> = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];
    let sums = sk_axis_sum(&input.view(), 1, 1);
    assert_eq!(sums, array![6.0, 15.0, 24.0]);
}

/// In-place scaling mutates the buffer and returns nothing.
#[test]
fn in_place_scale_mutates_without_copy() {
    let mut input: Array2<f64> = array![[1.0, 2.0], [3.0, 4.0]];
    sk_scale_in_place(&mut input, 3.0, 1);
    // No new array is returned; the input buffer is modified in place.
    assert_eq!(input, array![[3.0, 6.0], [9.0, 12.0]]);
}

/// The kernels stay generic over both supported floats.
#[test]
fn kernels_accept_either_float() {
    fn accepts<F: SKFloat>(input: &Array2<F>) {
        let _ = sk_axis_sum(&input.view(), 0, 1);
    }
    accepts(&array![[1.0_f32, 2.0]]);
    accepts(&array![[1.0_f64, 2.0]]);
}

// ---- Task 2.1 scenarios (spec `higher-order-kernels`, resolved parallelism) ----

/// Sequential path at `parallelism = 1`: the result is correct on a large array
/// and no parallel dispatch is incurred.
#[test]
fn sequential_path_at_parallelism_one_is_correct_on_large_array() {
    let n = 1 << 16;
    let input: Array1<f64> = Array1::from_shape_fn(n, |i| (i as f64) / 7.0);
    let output = sk_elementwise_transform(&input.view(), |x| 2.0 * x + 1.0, 1);
    assert_eq!(output.len(), n);
    // Spot-check a few elements to confirm correctness of the sequential path.
    assert!((output[0] - 1.0).abs() < 1e-12);
    assert!((output[n - 1] - (2.0 * input[n - 1] + 1.0)).abs() < 1e-12);
}

/// Elementwise parallel path matches the sequential reference within tolerance.
#[test]
fn elementwise_parallel_matches_sequential_on_large_array() {
    let n = 1 << 16;
    let input: Array1<f64> = Array1::from_shape_fn(n, |i| (i as f64) / 7.0);
    let f = |x: f64| (x * 3.0 + 2.0).sin();
    let sequential = sk_elementwise_transform(&input.view(), f, 1);
    let parallel = sk_elementwise_transform(&input.view(), f, 8);
    assert_close(&sequential, &parallel);
}

/// Binary-combine parallel path matches the sequential reference within tolerance.
#[test]
fn binary_parallel_matches_sequential_on_large_array() {
    let n = 1 << 16;
    let left: Array1<f64> = Array1::from_shape_fn(n, |i| (i as f64) / 5.0);
    let right: Array1<f64> = Array1::from_shape_fn(n, |i| ((i + 1) as f64) / 9.0);
    let f = |l: f64, r: f64| (l + r).atan();
    let sequential = sk_binary_combine(&left.view(), &right.view(), f, 1);
    let parallel = sk_binary_combine(&left.view(), &right.view(), f, 8);
    assert_close(&sequential, &parallel);
}

/// In-place scale parallel path matches the sequential reference.
#[test]
fn scale_parallel_matches_sequential_on_large_array() {
    let (rows, cols) = (512, 512);
    let mut sequential: Array2<f64> =
        Array2::from_shape_fn((rows, cols), |(i, j)| (i * cols + j) as f64);
    let mut parallel = sequential.clone();
    sk_scale_in_place(&mut sequential, 3.5, 1);
    sk_scale_in_place(&mut parallel, 3.5, 8);
    assert_eq!(sequential, parallel);
}

/// The same array shape driven through both plans agrees within tolerance
/// (no source-level duplication of the kernel).
#[test]
fn same_array_driven_through_both_plans_agrees() {
    let n = 1 << 15;
    let input: Array1<f64> = Array1::from_shape_fn(n, |i| (i as f64) * 0.25);
    let f = |x: f64| (x * 0.5).sin();
    let sequential = sk_elementwise_transform(&input.view(), f, 1);
    let parallel = sk_elementwise_transform(&input.view(), f, 4);
    assert_close(&sequential, &parallel);
}

// ---- Task 2.3 scenario (spec `higher-order-kernels`, race-free axis reduction) ----

/// A wide, tall matrix reduced along axis 0 with parallelism greater than one
/// equals the sequential column sums within tolerance. The parallel path must
/// accumulate per-chunk partials (no shared mutable accumulator), so this holds
/// across many rows without a race.
#[test]
fn parallel_axis_zero_sum_matches_sequential_reference() {
    let (rows, cols) = (4096, 256);
    let input: Array2<f64> = Array2::from_shape_fn((rows, cols), |(i, j)| {
        (i as f64) * 0.001 + (j as f64) * 0.5
    });
    let sequential = sk_axis_sum(&input.view(), 0, 1);
    let parallel = sk_axis_sum(&input.view(), 0, 8);
    assert_close(&sequential, &parallel);
}

// ---- Task 5.1 scenario (PRD §8.7 acceptance: concurrency under a shared pool) ----

/// Parallel kernels running from several threads on the shared rayon pool agree
/// with their sequential references, on small and large data.
#[test]
fn parallel_kernels_agree_under_concurrency() {
    let handles: Vec<_> = (0..4)
        .map(|_| {
            std::thread::spawn(|| {
                // Large data: elementwise + axis-0 reduction, parallel vs sequential.
                let n = 1 << 15;
                let input: Array1<f64> = Array1::from_shape_fn(n, |i| (i as f64) / 7.0);
                let seq_t = sk_elementwise_transform(&input.view(), |x| x.sin(), 1);
                let par_t = sk_elementwise_transform(&input.view(), |x| x.sin(), 8);
                assert_close(&seq_t, &par_t);

                let m: Array2<f64> =
                    Array2::from_shape_fn((512, 128), |(i, j)| (i as f64) * 0.001 + j as f64 * 0.25);
                let seq_s = sk_axis_sum(&m.view(), 0, 1);
                let par_s = sk_axis_sum(&m.view(), 0, 8);
                assert_close(&seq_s, &par_s);

                // Small data: sequential fallback stays correct under concurrency.
                let tiny: Array2<f64> = array![[1.0, 2.0], [3.0, 4.0]];
                assert_eq!(sk_axis_sum(&tiny.view(), 0, 1), array![4.0, 6.0]);
            })
        })
        .collect();
    for handle in handles {
        handle.join().expect("concurrent kernel thread must not panic");
    }
}