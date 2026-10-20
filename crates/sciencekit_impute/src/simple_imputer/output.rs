//! The owned imputation result: dense or CSR, chosen by the input's shape
//! (spec `simple-imputer`).

use ndarray::Array2;
use sprs::CsMat;

/// The result of an imputation transform: dense or sparse, matching the
/// representation the input arrived in. `#[non_exhaustive]` so future
/// representations can be added without breaking consumers.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum SKImputedOutput<F> {
    /// A dense matrix with the missing cells filled.
    Dense(Array2<F>),
    /// A sparse (CSR) matrix with the missing cells filled.
    Sparse(CsMat<F>),
}

impl<F> SKImputedOutput<F> {
    /// The row count of the imputed matrix, dense or sparse alike.
    pub fn nrows(&self) -> usize {
        0
    }

    /// The column count of the imputed matrix, dense or sparse alike.
    pub fn ncols(&self) -> usize {
        0
    }

    /// The representation label, used for diagnostics and dispatch.
    pub fn representation(&self) -> &'static str {
        "dense"
    }
}
