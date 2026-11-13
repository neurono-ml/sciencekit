//! Companion tests for Chan-Golub-LeVeque covariance merging on dense batches.

use ndarray::{Array2, Axis, array};
use sciencekit_common::SKBackendKind;
use sciencekit_math::backend::sk_math_backend_for_kind;

use super::accumulator::SKCovarianceAccumulator;
use super::merge::sk_update_dense;

#[test]
fn incremental_merge_hand_checked_two_batch_mean_and_scatter() {
    // Rows: [1, 2], [3, 6], [5, 10], [7, 14]
    // Mean = [4, 8]
    // Deviations: [-3, -6], [-1, -2], [1, 2], [3, 6]
    // Scatter = [[9+1+1+9, 18+2+2+18], [18+2+2+18, 36+4+4+36]] = [[20, 40], [40, 80]]
    let batch_one = array![[1.0_f64, 2.0], [3.0, 6.0]];
    let batch_two = array![[5.0_f64, 10.0], [7.0_f64, 14.0]];
    let backend = sk_math_backend_for_kind::<f64>(SKBackendKind::Faer);

    let mut accumulator = SKCovarianceAccumulator::<f64>::new(2);
    sk_update_dense(&mut accumulator, backend.as_ref(), batch_one.view(), 1);
    assert_eq!(accumulator.samples_seen(), 2);
    assert!((accumulator.mean()[0] - 2.0).abs() < 1e-12);
    assert!((accumulator.mean()[1] - 4.0).abs() < 1e-12);

    sk_update_dense(&mut accumulator, backend.as_ref(), batch_two.view(), 1);
    assert_eq!(accumulator.samples_seen(), 4);
    assert!((accumulator.mean()[0] - 4.0).abs() < 1e-12);
    assert!((accumulator.mean()[1] - 8.0).abs() < 1e-12);

    let expected_scatter = array![[20.0_f64, 40.0], [40.0, 80.0]];
    for (left, right) in accumulator.scatter().iter().zip(expected_scatter.iter()) {
        assert!((left - right).abs() < 1e-12, "left={left}, right={right}");
    }
}

#[test]
fn incremental_merge_two_batches_equal_one_batch_and_persist_state() {
    let data = Array2::from_shape_fn((37, 5), |(row, column)| {
        let base = (row as f64 + 1.0) * (column as f64 * 0.7 + 1.1);
        base + ((row * 5 + column * 3) % 7) as f64 * 0.35
    });
    let backend = sk_math_backend_for_kind::<f64>(SKBackendKind::Faer);

    let mut single = SKCovarianceAccumulator::<f64>::new(5);
    sk_update_dense(&mut single, backend.as_ref(), data.view(), 1);

    let mut two_batches = SKCovarianceAccumulator::<f64>::new(5);
    let first_half = data.slice_axis(Axis(0), ndarray::Slice::from(0..15));
    let second_half = data.slice_axis(Axis(0), ndarray::Slice::from(15..37));
    sk_update_dense(&mut two_batches, backend.as_ref(), first_half, 1);
    sk_update_dense(&mut two_batches, backend.as_ref(), second_half, 1);

    let mut persistent = SKCovarianceAccumulator::<f64>::new(5);
    let mut expected_count = 0usize;
    for chunk in data.axis_chunks_iter(Axis(0), 6) {
        let prior_count = persistent.samples_seen();
        sk_update_dense(&mut persistent, backend.as_ref(), chunk, 2);
        expected_count += chunk.nrows();
        assert_eq!(persistent.samples_seen(), prior_count + chunk.nrows());
    }
    assert_eq!(expected_count, 37);
    assert_eq!(two_batches.samples_seen(), single.samples_seen());
    assert_eq!(persistent.samples_seen(), single.samples_seen());

    for (left, right) in two_batches.mean().iter().zip(single.mean().iter()) {
        assert!((left - right).abs() < 1e-11);
    }
    for (left, right) in two_batches.scatter().iter().zip(single.scatter().iter()) {
        assert!((left - right).abs() < 1e-9);
    }
    for (left, right) in persistent.mean().iter().zip(single.mean().iter()) {
        assert!((left - right).abs() < 1e-11);
    }
    for (left, right) in persistent.scatter().iter().zip(single.scatter().iter()) {
        assert!((left - right).abs() < 1e-9);
    }
}
