//! The encoded-output enum dispatched by the builder's `sparse_output`.

use ndarray::Array2;
use sprs::CsMat;

/// The result of the encoding transform: dense or sparse, chosen by the
/// builder. `#[non_exhaustive]` so future representations can be added without
/// breaking consumers.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum SKEncodedOutput<F> {
    /// A dense indicator matrix.
    Dense(Array2<F>),
    /// A sparse (CSR) indicator matrix.
    Sparse(CsMat<F>),
}
