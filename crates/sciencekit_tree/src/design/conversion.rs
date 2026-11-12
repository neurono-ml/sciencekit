//! Sparse re-indexing and row-value lookup for the fit design.

use sciencekit_common::SKFloat;
use sprs::{CsMat, CsMatView, CsVecView};

/// Re-index a CSR view into CSC without densifying it.
///
/// The conversion counts the entries of every column, lays out the column
/// pointers and fills each column by walking the CSR rows in ascending order,
/// so the resulting column indices stay sorted.
pub(crate) fn sk_to_csc<F: SKFloat>(csr: &CsMatView<'_, F>) -> CsMat<F> {
    let (rows, columns) = (csr.rows(), csr.cols());
    let mut counts = vec![0usize; columns];
    for row in csr.outer_iterator() {
        for &column in row.indices() {
            counts[column] += 1;
        }
    }
    let mut indptr = Vec::with_capacity(columns + 1);
    indptr.push(0usize);
    for count in &counts {
        indptr.push(indptr[indptr.len() - 1] + count);
    }
    let mut indices = vec![0usize; csr.nnz()];
    let mut data = vec![F::zero(); csr.nnz()];
    let mut cursor = indptr.clone();
    for (row_index, row) in csr.outer_iterator().enumerate() {
        for (&column, &value) in row.indices().iter().zip(row.data().iter()) {
            indices[cursor[column]] = row_index;
            data[cursor[column]] = value;
            cursor[column] += 1;
        }
    }
    CsMat::new_csc((rows, columns), indptr, indices, data)
}

/// Read one stored feature value, treating an absent entry as zero.
pub(crate) fn sk_sparse_row_value<F: SKFloat>(row: &CsVecView<'_, F>, feature: usize) -> F {
    match row.indices().binary_search(&feature) {
        Ok(position) => row.data()[position],
        Err(_) => F::zero(),
    }
}
