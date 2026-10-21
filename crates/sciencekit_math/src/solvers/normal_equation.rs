//! The zero-copy sparse `AᵀA` normal-equation operator (spec
//! `iterative-linear-solvers`).

use ndarray::{ArrayView1, ArrayViewMut1, Axis};
use sciencekit_common::SKFloat;
use sprs::CsMatView;

use super::SKLinearOperator;

/// Zero-copy `Aᵀ A` operator over a borrowed CSR matrix.
///
/// The operator holds the sparse matrix by view and applies `AᵀA` to a vector
/// by walking the sparse structure in both directions: the forward `A v` reuses
/// the CSR×dense product, and the backward `Aᵀ w` scatters each stored entry.
/// The design is never densified and `AᵀA` is never formed.
pub struct SKCsrNormalEquationOperator<'a, F: SKFloat> {
    matrix: CsMatView<'a, F>,
}

impl<'a, F: SKFloat> SKCsrNormalEquationOperator<'a, F> {
    /// Build the operator over a borrowed CSR view (no copy is taken).
    pub fn new(matrix: CsMatView<'a, F>) -> Self {
        Self { matrix }
    }
}

impl<F: SKFloat> SKLinearOperator<F> for SKCsrNormalEquationOperator<'_, F> {
    fn dimension(&self) -> usize {
        self.matrix.cols()
    }

    fn apply_into(&self, input: ArrayView1<F>, mut output: ArrayViewMut1<F>) {
        assert_eq!(
            input.len(),
            self.matrix.cols(),
            "input length must match the operator dimension"
        );
        // Forward product `A v`, reused as a single dense column.
        let column = input.insert_axis(Axis(1));
        let product = crate::sparse_ops::sk_csr_dense_product(&self.matrix, &column);
        // Backward product `Aᵀ w`: scatter each stored entry into the output.
        output.fill(F::zero());
        for (row_index, sparse_row) in self.matrix.outer_iterator().enumerate() {
            let weight = product[[row_index, 0]];
            for (&column_index, &value) in sparse_row.indices().iter().zip(sparse_row.data()) {
                output[column_index] = output[column_index] + value * weight;
            }
        }
    }
}
