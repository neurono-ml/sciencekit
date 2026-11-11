//! The dense products consumed by the randomized range finder.
//!
//! The operator wraps the input representation and dispatches **once** per
//! operation. This first step covers the dense representation only; the sparse
//! variant is added with the sparse path of the kernel.

use ndarray::{Array2, ArrayView2};
use sciencekit_common::{SKDataView, SKError, SKFloat};

use crate::backend::SKMathBackend;

/// A matrix operand that keeps its dense or sparse representation.
pub(crate) enum SKMatrixOperator<'a, F: SKFloat> {
    /// A dense, row-major feature matrix.
    Dense(ArrayView2<'a, F>),
}

impl<'a, F: SKFloat> SKMatrixOperator<'a, F> {
    /// Wraps a feature view, dispatching once on the representation.
    ///
    /// A representation added to [`SKDataView`] in the future is rejected
    /// with an explicit error instead of being silently densified.
    pub(crate) fn new(view: SKDataView<'a, F>) -> Result<Self, SKError> {
        match view {
            SKDataView::Dense(dense) => Ok(SKMatrixOperator::Dense(dense)),
            _ => Err(SKError::UnsupportedRepresentation {
                representation: "csr",
                suggestion: "use the dense representation of the feature matrix",
            }),
        }
    }

    /// The number of rows (observations).
    pub(crate) fn rows(&self) -> usize {
        match self {
            SKMatrixOperator::Dense(dense) => dense.nrows(),
        }
    }

    /// The number of columns (features).
    pub(crate) fn columns(&self) -> usize {
        match self {
            SKMatrixOperator::Dense(dense) => dense.ncols(),
        }
    }

    /// Computes `X · right`, with `right` a dense `n × p` matrix.
    ///
    /// The GEMM runs with a single worker so the result does not depend on the
    /// thread count of the machine.
    pub(crate) fn multiply(
        &self,
        backend: &dyn SKMathBackend<F>,
        right: &ArrayView2<'_, F>,
    ) -> Array2<F> {
        match self {
            SKMatrixOperator::Dense(dense) => backend.gemm(*dense, *right, F::one(), 1),
        }
    }

    /// Computes `Xᵀ · right`, with `right` a dense `m × p` matrix.
    pub(crate) fn transpose_multiply(
        &self,
        backend: &dyn SKMathBackend<F>,
        right: &ArrayView2<'_, F>,
    ) -> Array2<F> {
        match self {
            SKMatrixOperator::Dense(dense) => {
                let transposed = dense.t().to_owned();
                backend.gemm(transposed.view(), *right, F::one(), 1)
            }
        }
    }
}
