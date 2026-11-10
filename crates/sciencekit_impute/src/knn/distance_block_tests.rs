//! Companion tests for the fixed-size distance blocks
//! (spec `nearest-neighbor-imputation`).

use ndarray::Array2;
use sciencekit_math::sk_nan_euclidean_distance_matrix;

use crate::knn::distance_block::{
    SK_DISTANCE_BLOCK_BYTES, sk_distance_block, sk_distance_block_ranges, sk_distance_block_rows,
};

const SCALAR_BYTES: usize = 8;

#[test]
fn knn_distance_block_rows_stays_within_the_byte_budget() {
    for training_rows in [1_usize, 10, 1_000, 100_000] {
        let block_rows = sk_distance_block_rows(training_rows, SCALAR_BYTES);
        assert!(block_rows >= 1);
        assert!(
            block_rows * training_rows * SCALAR_BYTES <= SK_DISTANCE_BLOCK_BYTES || block_rows == 1,
            "a block of {block_rows} × {training_rows} scalars must fit the budget"
        );
    }
}

#[test]
fn knn_distance_block_rows_is_minimum_one_for_an_empty_training_set() {
    assert_eq!(sk_distance_block_rows(0, SCALAR_BYTES), 1);
}

#[test]
fn knn_distance_block_ranges_partition_every_receiver_row() {
    let ranges = sk_distance_block_ranges(10, 4);
    assert_eq!(ranges.len(), 3);
    assert_eq!(ranges[0], 0..4);
    assert_eq!(ranges[1], 4..8);
    assert_eq!(ranges[2], 8..10);
    let covered: usize = ranges.iter().map(|range| range.len()).sum();
    assert_eq!(covered, 10);
}

#[test]
fn knn_distance_block_ranges_handle_a_single_receiver() {
    assert_eq!(sk_distance_block_ranges(1, 512), vec![0..1]);
    assert_eq!(sk_distance_block_ranges(0, 512), Vec::new());
}

#[test]
fn knn_distance_block_matches_the_full_nan_aware_matrix() {
    let receivers = Array2::from_shape_fn((3, 2), |(row, column)| {
        if (row + column) % 3 == 0 {
            f64::NAN
        } else {
            row as f64 + column as f64 / 2.0
        }
    });
    let training = Array2::from_shape_fn((5, 2), |(row, column)| {
        if (row * column) % 4 == 0 {
            f64::NAN
        } else {
            row as f64 * 0.5 - column as f64
        }
    });
    let block = sk_distance_block(&receivers.view(), &training.view());
    let full = sk_nan_euclidean_distance_matrix(&receivers.view(), &training.view());
    assert_eq!(block.dim(), full.dim());
    for (found, expected) in block.iter().zip(full.iter()) {
        assert!(
            (found - expected).abs() < 1e-12 || (found.is_nan() && expected.is_nan()),
            "expected {expected}, found {found}"
        );
    }
}

#[test]
fn knn_distance_block_handles_a_single_receiver_and_a_large_receiver_count() {
    let training = Array2::from_shape_fn((40, 3), |(row, column)| {
        ((row * 7 + column * 3) % 17) as f64 / 4.0
    });
    let single = Array2::from_shape_fn((1, 3), |(_, column)| column as f64);
    let single_distance = sk_distance_block(&single.view(), &training.view());
    assert_eq!(single_distance.dim(), (1, 40));

    let many = Array2::from_shape_fn((5_000, 3), |(row, column)| {
        ((row * 11 + column * 5) % 29) as f64 / 8.0
    });
    let block_rows = sk_distance_block_rows(training.nrows(), SCALAR_BYTES);
    let mut processed = 0;
    for range in sk_distance_block_ranges(many.nrows(), block_rows) {
        let block = many.slice(ndarray::s![range.clone(), ..]);
        let distance = sk_distance_block(&block, &training.view());
        assert_eq!(distance.dim(), (range.len(), training.nrows()));
        assert!(distance.iter().all(|value| value.is_finite()));
        processed += range.len();
    }
    assert_eq!(processed, many.nrows());
}
