//! The owned mask: dense or CSR, chosen by the input representation
//! (spec `missing-indicator`).

use ndarray::Array2;
use sprs::CsMat;

/// The result of a missingness-mask transform: dense or sparse, matching the
/// representation the input arrived in (or the requested sparse output).
/// `#[non_exhaustive]` so future representations can be added without breaking
/// consumers.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum SKIndicatorOutput<F> {
    /// A dense mask with one column per feature in the fitted scope.
    Dense(Array2<F>),
    /// A sparse (CSR) mask whose stored entries are exactly the missing cells.
    Sparse(CsMat<F>),
}

impl<F> SKIndicatorOutput<F> {
    /// The row count of the mask, dense or sparse alike.
    pub fn nrows(&self) -> usize {
        match self {
            SKIndicatorOutput::Dense(matrix) => matrix.nrows(),
            SKIndicatorOutput::Sparse(matrix) => matrix.shape().0,
        }
    }

    /// The column count of the mask, dense or sparse alike.
    pub fn ncols(&self) -> usize {
        match self {
            SKIndicatorOutput::Dense(matrix) => matrix.ncols(),
            SKIndicatorOutput::Sparse(matrix) => matrix.shape().1,
        }
    }

    /// The representation label, used for diagnostics and dispatch.
    pub fn representation(&self) -> &'static str {
        match self {
            SKIndicatorOutput::Dense(_) => "dense",
            SKIndicatorOutput::Sparse(_) => "csr",
        }
    }
}
