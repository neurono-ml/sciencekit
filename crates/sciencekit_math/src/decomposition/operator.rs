//! The dense and sparse products consumed by the randomized range finder.
//!
//! The operator wraps the input representation and dispatches **once** per
//! operation: a dense input uses the backend GEMM, a sparse input walks the
//! CSR structure. No sparse input is ever densified.

use ndarray::{Array2, ArrayView2};
use sciencekit_common::{SKDataView, SKError, SKFloat};
use sprs::CsMatView;

use crate::backend::SKMathBackend;

#[cfg(test)]
use super::instrumentation::sk_record_dense_intermediate;
use crate::sparse_ops::sk_csr_dense_product;

/// A matrix operand that keeps its dense or sparse representation.
pub(crate) enum SKMatrixOperator<'a, F: SKFloat> {
    /// A dense, row-major feature matrix.
    Dense(ArrayView2<'a, F>),
    /// A sparse compressed-row feature matrix.
    Sparse(CsMatView<'a, F>),
}

impl<'a, F: SKFloat> SKMatrixOperator<'a, F> {
    /// Wraps a feature view, dispatching once on the representation.
    ///
    /// A representation added to [`SKDataView`] in the future is rejected
    /// with an explicit error instead of being silently densified.
    pub(crate) fn new(view: SKDataView<'a, F>) -> Result<Self, SKError> {
        match view {
            SKDataView::Dense(dense) => Ok(SKMatrixOperator::Dense(dense)),
            SKDataView::Sparse(sparse) => Ok(SKMatrixOperator::Sparse(sparse)),
            _ => Err(SKError::UnsupportedRepresentation {
                representation: "unknown",
                suggestion: "use the dense or the sparse representation of the feature matrix",
            }),
        }
    }

    /// The number of rows (observations).
    pub(crate) fn rows(&self) -> usize {
        match self {
            SKMatrixOperator::Dense(dense) => dense.nrows(),
            SKMatrixOperator::Sparse(sparse) => sparse.rows(),
        }
    }

    /// The number of columns (features).
    pub(crate) fn columns(&self) -> usize {
        match self {
            SKMatrixOperator::Dense(dense) => dense.ncols(),
            SKMatrixOperator::Sparse(sparse) => sparse.cols(),
        }
    }

    /// Computes `X · right`, with `right` a dense `n × p` matrix.
    ///
    /// The sparse path walks the CSR structure and never materializes a dense
    /// copy of the input. The GEMM runs with a single worker so the result does
    /// not depend on the thread count of the machine.
    pub(crate) fn multiply(
        &self,
        backend: &dyn SKMathBackend<F>,
        right: &ArrayView2<'_, F>,
    ) -> Array2<F> {
        match self {
            SKMatrixOperator::Dense(dense) => {
                let output = backend.gemm(*dense, *right, F::one(), 1);
                #[cfg(test)]
                sk_record_dense_intermediate(output.len());
                output
            }
            SKMatrixOperator::Sparse(sparse) => {
                let output = sk_csr_dense_product(sparse, right);
                #[cfg(test)]
                sk_record_dense_intermediate(output.len());
                output
            }
        }
    }

    /// Computes `Xᵀ · right`, with `right` a dense `m × p` matrix.
    ///
    /// The sparse path accumulates the transposed product row by row over the
    /// CSR structure; the dense path transposes the operand and multiplies.
    pub(crate) fn transpose_multiply(
        &self,
        backend: &dyn SKMathBackend<F>,
        right: &ArrayView2<'_, F>,
    ) -> Array2<F> {
        match self {
            SKMatrixOperator::Dense(dense) => {
                let transposed = dense.t().to_owned();
                let output = backend.gemm(transposed.view(), *right, F::one(), 1);
                #[cfg(test)]
                sk_record_dense_intermediate(output.len());
                output
            }
            SKMatrixOperator::Sparse(sparse) => {
                let mut output = Array2::<F>::zeros((sparse.cols(), right.ncols()));
                for (row_index, sparse_row) in sparse.outer_iterator().enumerate() {
                    let right_row = right.row(row_index);
                    for (&column, &value) in
                        sparse_row.indices().iter().zip(sparse_row.data().iter())
                    {
                        let mut output_row = output.row_mut(column);
                        for (accumulator, &factor) in output_row.iter_mut().zip(right_row.iter()) {
                            *accumulator = *accumulator + value * factor;
                        }
                    }
                }
                #[cfg(test)]
                sk_record_dense_intermediate(output.len());
                output
            }
        }
    }
}
