//! Per-column donor bookkeeping for the nearest-neighbor imputer
//! (spec `nearest-neighbor-imputation`).

use ndarray::ArrayView2;

/// The training rows that observed each column, in ascending row order.
///
/// A row that is missing the column is never a donor for it. An all-missing
/// column contributes no donor at all.
pub(super) fn sk_column_donors(
    missing_mask: &ArrayView2<'_, bool>,
) -> Vec<Vec<usize>> {
    let (rows_count, columns_count) = missing_mask.dim();
    let mut donors = vec![Vec::new(); columns_count];
    for row in 0..rows_count {
        for column in 0..columns_count {
            if !missing_mask[[row, column]] {
                donors[column].push(row);
            }
        }
    }
    donors
}
