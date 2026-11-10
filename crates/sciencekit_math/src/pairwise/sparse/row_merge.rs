//! Sorted-row merge helper shared by the sparse kernels.
//!
//! Rows are canonical CSR rows: stored column indices are ascending and
//! duplicate-free. Merging two of them classifies every coordinate as shared,
//! left-only or right-only, so a kernel visits exactly the stored entries and
//! treats an implicit coordinate as the observed value zero.

use sciencekit_common::SKFloat;

/// One coordinate produced by merging two sorted sparse rows.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum SKMergedCoordinate<F> {
    /// The coordinate is stored in both rows.
    Shared {
        /// The shared column index.
        column: usize,
        /// The value stored in the left row.
        left_value: F,
        /// The value stored in the right row.
        right_value: F,
    },
    /// The coordinate is stored only in the left row.
    LeftOnly {
        /// The column index.
        column: usize,
        /// The value stored in the left row.
        left_value: F,
    },
    /// The coordinate is stored only in the right row.
    RightOnly {
        /// The column index.
        column: usize,
        /// The value stored in the right row.
        right_value: F,
    },
}

/// Merge two ascending duplicate-free sparse rows coordinate by coordinate.
///
/// The returned iterator yields coordinates in ascending column order and runs
/// in `O(nnz_left + nnz_right)`.
pub(crate) fn sk_merge_sorted_rows<'a, F: SKFloat>(
    left_indices: &'a [usize],
    left_values: &'a [F],
    right_indices: &'a [usize],
    right_values: &'a [F],
) -> SKRowMerge<'a, F> {
    debug_assert_eq!(left_indices.len(), left_values.len());
    debug_assert_eq!(right_indices.len(), right_values.len());
    SKRowMerge {
        left_indices,
        left_values,
        right_indices,
        right_values,
        left_position: 0,
        right_position: 0,
    }
}

/// The iterator produced by [`sk_merge_sorted_rows`].
pub(crate) struct SKRowMerge<'a, F> {
    left_indices: &'a [usize],
    left_values: &'a [F],
    right_indices: &'a [usize],
    right_values: &'a [F],
    left_position: usize,
    right_position: usize,
}

impl<F: SKFloat> Iterator for SKRowMerge<'_, F> {    type Item = SKMergedCoordinate<F>;

    fn next(&mut self) -> Option<Self::Item> {
        let left = self
            .left_indices
            .get(self.left_position)
            .map(|&column| (column, self.left_values[self.left_position]));
        let right = self
            .right_indices
            .get(self.right_position)
            .map(|&column| (column, self.right_values[self.right_position]));
        match (left, right) {
            (None, None) => None,
            (Some((column, left_value)), None) => {
                self.left_position += 1;
                Some(SKMergedCoordinate::LeftOnly {
                    column,
                    left_value,
                })
            }
            (None, Some((column, right_value))) => {
                self.right_position += 1;
                Some(SKMergedCoordinate::RightOnly {
                    column,
                    right_value,
                })
            }
            (Some((left_column, left_value)), Some((right_column, right_value))) => {
                if left_column == right_column {
                    self.left_position += 1;
                    self.right_position += 1;
                    Some(SKMergedCoordinate::Shared {
                        column: left_column,
                        left_value,
                        right_value,
                    })
                } else if left_column < right_column {
                    self.left_position += 1;
                    Some(SKMergedCoordinate::LeftOnly {
                        column: left_column,
                        left_value,
                    })
                } else {
                    self.right_position += 1;
                    Some(SKMergedCoordinate::RightOnly {
                        column: right_column,
                        right_value,
                    })
                }
            }
        }
    }
}

/// Dot product of two sorted sparse rows over their shared coordinates.
///
/// One-sided coordinates multiply against the implicit zero, so only the merge's
/// shared entries contribute.
pub(crate) fn sk_sparse_dot_product<F: SKFloat>(
    left_indices: &[usize],
    left_values: &[F],
    right_indices: &[usize],
    right_values: &[F],
) -> F {
    sk_merge_sorted_rows(left_indices, left_values, right_indices, right_values).fold(
        F::zero(),
        |accumulator, coordinate| match coordinate {
            SKMergedCoordinate::Shared {
                left_value,
                right_value,
                ..
            } => accumulator + left_value * right_value,
            _ => accumulator,
        },
    )
}
