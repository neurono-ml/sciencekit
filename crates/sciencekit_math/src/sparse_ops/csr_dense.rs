//! Sparse (`sprs`) product kernels: sparse (CSR) × dense.

use ndarray::{Array, Array2, ArrayView2, azip};
use sciencekit_common::SKFloat;
use sprs::CsMatView;

/// Product of a CSR matrix and a dense matrix, as a dense result.
///
/// Walks each non-zero entry of the CSR operand once (never densifying it),
/// accumulating `value * dense_row` into the matching output row via [`azip!`].
pub fn sk_csr_dense_product<F: SKFloat>(csr: &CsMatView<F>, dense: &ArrayView2<F>) -> Array2<F> {
    assert_eq!(
        csr.cols(),
        dense.nrows(),
        "CSR column count must match dense row count"
    );
    let mut output = Array::zeros((csr.rows(), dense.ncols()));
    for (row_index, sparse_row) in csr.outer_iterator().enumerate() {
        let mut output_row = output.row_mut(row_index);
        for (&column, &value) in sparse_row.indices().iter().zip(sparse_row.data().iter()) {
            let dense_row = dense.row(column);
            azip!((acc in &mut output_row, d in dense_row) {
                *acc = *acc + value * *d;
            });
        }
    }
    output
}
