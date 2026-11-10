//! TDD tests for the sorted-row merge helper (spec `pairwise-distances`).

use super::row_merge::{SKMergedCoordinate, sk_merge_sorted_rows};

/// Collect a merge into `(column, classification)` tuples for assertions.
fn merge_labels(
    left_indices: &[usize],
    left_values: &[f64],
    right_indices: &[usize],
    right_values: &[f64],
) -> Vec<(usize, &'static str)> {
    sk_merge_sorted_rows(left_indices, left_values, right_indices, right_values)
        .map(|coordinate| match coordinate {
            SKMergedCoordinate::Shared { column, .. } => (column, "shared"),
            SKMergedCoordinate::LeftOnly { column, .. } => (column, "left"),
            SKMergedCoordinate::RightOnly { column, .. } => (column, "right"),
        })
        .collect()
}

/// Overlapping, left-only and right-only coordinates are classified correctly.
#[test]
fn sparse_row_merge_classifies_shared_and_one_sided_coordinates() {
    let labels = merge_labels(
        &[0, 2, 5],
        &[1.0, 2.0, 3.0],
        &[1, 2, 3],
        &[4.0, 5.0, 6.0],
    );
    assert_eq!(
        labels,
        vec![
            (0, "left"),
            (1, "right"),
            (2, "shared"),
            (3, "right"),
            (5, "left"),
        ]
    );
}

/// Shared coordinates carry both stored values unchanged.
#[test]
fn sparse_row_merge_pairs_shared_values() {
    let mut merge =
        sk_merge_sorted_rows(&[1, 4], &[1.5_f64, -2.0], &[1, 4, 7], &[0.5, 8.0, 3.0]);
    match merge.next().expect("first coordinate") {
        SKMergedCoordinate::Shared {
            column,
            left_value,
            right_value,
        } => {
            assert_eq!(column, 1);
            assert!((left_value - 1.5).abs() < 1e-12);
            assert!((right_value - 0.5).abs() < 1e-12);
        }
        other => panic!("expected a shared coordinate, got {other:?}"),
    }
    match merge.next().expect("second coordinate") {
        SKMergedCoordinate::Shared {
            column,
            left_value,
            right_value,
        } => {
            assert_eq!(column, 4);
            assert!((left_value + 2.0).abs() < 1e-12);
            assert!((right_value - 8.0).abs() < 1e-12);
        }
        other => panic!("expected a shared coordinate, got {other:?}"),
    }
    match merge.next().expect("third coordinate") {
        SKMergedCoordinate::RightOnly { column, right_value } => {
            assert_eq!(column, 7);
            assert!((right_value - 3.0).abs() < 1e-12);
        }
        other => panic!("expected a right-only coordinate, got {other:?}"),
    }
    assert!(merge.next().is_none());
}

/// Two empty rows yield no coordinates.
#[test]
fn sparse_row_merge_two_empty_rows_is_empty() {
    assert!(merge_labels(&[], &[], &[], &[]).is_empty());
}

/// An empty left row yields only right-only coordinates.
#[test]
fn sparse_row_merge_empty_left_row_is_right_only() {
    let labels = merge_labels(&[], &[], &[0, 9], &[1.0, 2.0]);
    assert_eq!(labels, vec![(0, "right"), (9, "right")]);
}

/// An empty right row yields only left-only coordinates.
#[test]
fn sparse_row_merge_empty_right_row_is_left_only() {
    let labels = merge_labels(&[3, 4], &[1.0, 2.0], &[], &[]);
    assert_eq!(labels, vec![(3, "left"), (4, "left")]);
}

/// Canonical duplicate-free ascending rows produce strictly ascending output.
#[test]
fn sparse_row_merge_keeps_columns_ascending() {
    let labels = merge_labels(
        &[0, 1, 4, 6],
        &[1.0, 1.0, 1.0, 1.0],
        &[1, 2, 4, 5, 6],
        &[1.0, 1.0, 1.0, 1.0, 1.0],
    );
    let columns: Vec<usize> = labels.iter().map(|(column, _)| *column).collect();
    assert_eq!(columns, vec![0, 1, 2, 4, 5, 6]);
    assert!(columns.windows(2).all(|pair| pair[0] < pair[1]));
}
