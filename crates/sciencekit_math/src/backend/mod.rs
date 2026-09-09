//! The BLAS/LAPACK interface (spec `blas-interface`, PRD §6.2).
//!
//! A single host-centric [`SKMathBackend`] abstraction routes the heavy dense
//! algebra (GEMM, SVD, QR, Cholesky, LU, eigendecomposition, least squares,
//! norms). The trait is generic over the sealed [`SKFloat`] scalar bound,
//! accepts zero-copy ndarray views and returns owned ndarray results — no
//! concrete backend type leaks into this surface.
//!
//! The pure-Rust default is [`SKFaerBackend`] (wave-plan-foundation Decision 1,
//! MSRV 1.85); the opt-in `blas-backend` feature swaps it for
//! `ndarray-linalg` over the configured `blas-src` backend, while faer stays
//! available as a fallback.

mod decompositions;
mod dispatch;
mod faer_backend;
mod kernel;

#[cfg(feature = "blas-backend")]
mod ndarray_backend;

pub use decompositions::{
    SKLUDecomposition, SKLeastSquaresSolution, SKQRDecomposition, SKSingularValueDecomposition,
};
pub use dispatch::sk_default_math_backend;
pub use faer_backend::{SKFaerBackend, SKMatrixMultiplyBackend};
pub use kernel::{SKMathBackend, SKNormKind};

#[cfg(feature = "blas-backend")]
pub use ndarray_backend::SKNdArrayLinalgBackend;

#[cfg(test)]
mod backend_tests;