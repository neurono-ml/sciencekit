//! Sparse CSR gathering into CSC subsamples and zero-copy sparse row lookup.

use ndarray::{Array2, ArrayView2};
use sciencekit_common::SKFloat;
use sprs::{CsMat, CsMatView, CsVecView};

use super::subsample::IsolationSampleSource;

/// Owned CSC subsample built directly from selected CSR rows without densification.
pub(crate) struct SparseSubsample<F> {
    csc: CsMat<F>,
}

impl<F: SKFloat> SparseSubsample<F> {
    /// Gathers `sampled_rows` from `csr` into a compact CSC subsample.
    pub(crate) fn from_csr_rows(csr: CsMatView<'_, F>, sampled_rows: &[usize]) -> Self {
        let column_count = csr.cols();
        let mut column_counts = vec![0usize; column_count];
        sampled_rows.iter().for_each(|&source_row| {
            let row_view = csr.outer_view(source_row).expect("row within bounds");
            row_view
                .indices()
                .iter()
                .for_each(|&column_index| column_counts[column_index] += 1);
        });

        let mut column_pointers = Vec::with_capacity(column_count + 1);
        column_pointers.push(0usize);
        column_counts.iter().for_each(|&count| {
            let previous = *column_pointers.last().expect("non-empty pointers");
            column_pointers.push(previous + count);
        });
        let total_stored = *column_pointers.last().expect("non-empty pointers");

        let mut row_indices = vec![0usize; total_stored];
        let mut values = vec![F::zero(); total_stored];
        let mut cursors = column_pointers.clone();
        sampled_rows
            .iter()
            .enumerate()
            .for_each(|(local_row, &source_row)| {
                let row_view = csr.outer_view(source_row).expect("row within bounds");
                row_view
                    .indices()
                    .iter()
                    .zip(row_view.data().iter())
                    .for_each(|(&column_index, &value)| {
                        let slot = cursors[column_index];
                        row_indices[slot] = local_row;
                        values[slot] = value;
                        cursors[column_index] = slot + 1;
                    });
            });

        Self {
            csc: CsMat::new_csc(
                (sampled_rows.len(), column_count),
                column_pointers,
                row_indices,
                values,
            ),
        }
    }

    /// Number of stored entries in the CSC subsample (for non-densification checks).
    #[cfg(test)]
    pub(crate) fn stored_entry_count(&self) -> usize {
        self.csc.nnz()
    }
}

impl<F: SKFloat> IsolationSampleSource<F> for SparseSubsample<F> {
    fn sample_count(&self) -> usize {
        self.csc.rows()
    }

    fn feature_count(&self) -> usize {
        self.csc.cols()
    }

    fn feature_bounds(&self, rows: &[usize], feature_index: usize) -> (F, F) {
        let column = self
            .csc
            .outer_view(feature_index)
            .expect("column within bounds");
        let mut bounds: Option<(F, F)> = None;
        for_each_column_value(column, rows, |_, value| {
            bounds = Some(match bounds {
                None => (value, value),
                Some((current_minimum, current_maximum)) => {
                    let next_minimum = if value < current_minimum {
                        value
                    } else {
                        current_minimum
                    };
                    let next_maximum = if value > current_maximum {
                        value
                    } else {
                        current_maximum
                    };
                    (next_minimum, next_maximum)
                }
            });
        });
        bounds.unwrap_or_else(|| (F::zero(), F::zero()))
    }

    fn partition_rows(
        &self,
        rows: &[usize],
        feature_index: usize,
        threshold: F,
    ) -> (Vec<usize>, Vec<usize>) {
        let column = self
            .csc
            .outer_view(feature_index)
            .expect("column within bounds");
        let mut left_rows = Vec::new();
        let mut right_rows = Vec::new();
        for_each_column_value(column, rows, |row_index, value| {
            if value <= threshold {
                left_rows.push(row_index);
            } else {
                right_rows.push(row_index);
            }
        });
        (left_rows, right_rows)
    }
}

fn for_each_column_value<F: SKFloat>(
    column: CsVecView<'_, F>,
    sorted_rows: &[usize],
    mut visit: impl FnMut(usize, F),
) {
    let mut stored = column
        .indices()
        .iter()
        .copied()
        .zip(column.data().iter().copied())
        .peekable();
    sorted_rows.iter().copied().for_each(|row_index| {
        while let Some(&(stored_row, _)) = stored.peek() {
            if stored_row < row_index {
                stored.next();
            } else {
                break;
            }
        }
        let value = match stored.peek() {
            Some(&(stored_row, stored_value)) if stored_row == row_index => {
                stored.next();
                stored_value
            }
            _ => F::zero(),
        };
        visit(row_index, value);
    });
}

/// Gathers `sampled_rows` from a dense matrix into an owned subsample matrix.
pub(crate) fn sk_gather_dense_rows<F: SKFloat>(
    matrix: ArrayView2<'_, F>,
    sampled_rows: &[usize],
) -> Array2<F> {
    Array2::from_shape_fn(
        (sampled_rows.len(), matrix.ncols()),
        |(local_row, column)| matrix[[sampled_rows[local_row], column]],
    )
}

/// Looks up one feature in a sparse CSR row view, treating absent entries as zero.
pub(crate) fn sk_sparse_row_value<F: SKFloat>(row: CsVecView<'_, F>, feature_index: usize) -> F {
    row.indices()
        .binary_search(&feature_index)
        .map_or_else(|_| F::zero(), |position| row.data()[position])
}
