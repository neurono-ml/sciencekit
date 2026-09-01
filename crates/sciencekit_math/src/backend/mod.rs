//! The BLAS/LAPACK interface (spec `blas-interface`, PRD §6.2).
//!
//! A single [`SKMathBackend`] abstraction routes the heavy dense algebra
//! (GEMM, SVD, QR, Cholesky). The pure-Rust default is [`SKFaerBackend`]
//! (wave-plan-foundation Decision 1, MSRV 1.85); the opt-in `blas-backend`
//! feature swaps it for `ndarray-linalg` over the configured `blas-src`
//! backend, while faer stays available as a fallback.

mod faer_backend;

#[cfg(feature = "blas-backend")]
mod ndarray_backend;

use faer::{Mat, MatRef};
use sciencekit_common::SKError;

pub use faer_backend::{SKFaerBackend, SKMatrixMultiplyBackend};

#[cfg(feature = "blas-backend")]
pub use ndarray_backend::SKNdArrayLinalgBackend;

/// The result of a singular value decomposition `A = U Σ Vᵀ`.
#[derive(Debug, Clone)]
pub struct SKSingularValueDecomposition {
    /// The left singular vectors `U`.
    pub u: Mat<f64>,
    /// The singular values, non-increasing.
    pub singular_values: Vec<f64>,
    /// The right singular vectors `V` (columns are `Vᵀ` rows).
    pub v: Mat<f64>,
}

/// The result of a thin QR decomposition `A = Q R`.
#[derive(Debug, Clone)]
pub struct SKQRDecomposition {
    /// The orthogonal factor `Q`.
    pub q: Mat<f64>,
    /// The upper-triangular factor `R`.
    pub r: Mat<f64>,
}

/// The dense linear-algebra backend abstraction.
///
/// Implementations are pure and deterministic, and must be `Send + Sync` so an
/// execution plan can dispatch heavy algebra across threads.
pub trait SKMathBackend: Send + Sync {
    /// The backend name, recorded in observability spans.
    fn name(&self) -> &'static str;

    /// Compute `C = α · A·B` for dense `A (m×k)` and `B (k×n)`.
    fn gemm(&self, a: MatRef<f64>, b: MatRef<f64>, alpha: f64) -> Mat<f64>;

    /// Compute the singular value decomposition of `A`.
    fn svd(&self, a: MatRef<f64>) -> Result<SKSingularValueDecomposition, SKError>;

    /// Compute the (thin) QR decomposition of `A`.
    fn qr(&self, a: MatRef<f64>) -> SKQRDecomposition;

    /// Compute the Cholesky factor `L` of a positive-definite `A` (`A = L Lᵀ`).
    fn cholesky(&self, a: MatRef<f64>) -> Result<Mat<f64>, SKError>;
}

/// The default math backend for the current build configuration.
///
/// With the `blas-backend` feature enabled this returns the `ndarray-linalg`
/// backend; otherwise the pure-Rust [`SKFaerBackend`]. Execution planning (Phase
/// 0.4) calls this to route dense algebra.
pub fn sk_default_math_backend() -> Box<dyn SKMathBackend> {
    #[cfg(feature = "blas-backend")]
    {
        Box::new(SKNdArrayLinalgBackend::new())
    }
    #[cfg(not(feature = "blas-backend"))]
    {
        Box::new(SKFaerBackend::new())
    }
}

#[cfg(test)]
mod backend_tests;
