//! The opt-in `blas-backend` backend backed by `ndarray-linalg` over the
//! configured `blas-src` library (OpenBLAS/BLIS/MKL/Accelerate).
//!
//! Gated behind the `blas-backend` feature, which is never enabled by default.
//! When enabled, heavy dense algebra resolves to this backend; the pure-Rust
//! [`SKFaerBackend`] remains available as a fallback.

use faer::{Mat, MatRef, Unbind};
use ndarray::{Array2, ArrayView2};
use ndarray_linalg::{Cholesky, QR, SVD, UPLO};
use sciencekit_common::SKError;

use super::{SKMathBackend, SKQRDecomposition, SKSingularValueDecomposition};

/// The system-BLAS backend over `ndarray-linalg` + `blas-src`.
#[derive(Debug, Clone, Copy, Default)]
pub struct SKNdArrayLinalgBackend;

impl SKNdArrayLinalgBackend {
    /// Create the backend.
    pub fn new() -> Self {
        SKNdArrayLinalgBackend
    }
}

impl SKMathBackend for SKNdArrayLinalgBackend {
    fn gemm(&self, a: MatRef<f64>, b: MatRef<f64>, alpha: f64) -> Mat<f64> {
        let a_dense = to_ndarray(a);
        let b_dense = to_ndarray(b);
        let product = a_dense.dot(&b_dense);
        from_ndarray(&(product * alpha).view())
    }

    fn svd(&self, a: MatRef<f64>) -> Result<SKSingularValueDecomposition, SKError> {
        let a_dense = to_ndarray(a);
        let (u, singular_values, vh) = a_dense
            .svd(true, true)
            .map_err(|error| SKError::Conversion(error.to_string()))?;
        let u = u.ok_or_else(|| SKError::Conversion("SVD did not return U".into()))?;
        let vh = vh.ok_or_else(|| SKError::Conversion("SVD did not return V".into()))?;
        Ok(SKSingularValueDecomposition {
            u: from_ndarray(&u.view()),
            singular_values: singular_values.to_vec(),
            // faer stores `V`; ndarray-linalg returns `Vᵀ`, so transpose back.
            v: from_ndarray(&vh.t()),
        })
    }

    fn qr(&self, a: MatRef<f64>) -> SKQRDecomposition {
        let a_dense = to_ndarray(a);
        let (q, r) = a_dense.qr().expect("ndarray-linalg QR returned an error");
        SKQRDecomposition {
            q: from_ndarray(&q.view()),
            r: from_ndarray(&r.view()),
        }
    }

    fn cholesky(&self, a: MatRef<f64>) -> Result<Mat<f64>, SKError> {
        let a_dense = to_ndarray(a);
        let factor = a_dense
            .cholesky(UPLO::Lower)
            .map_err(|error| SKError::Conversion(error.to_string()))?;
        Ok(from_ndarray(&factor.view()))
    }
}

/// Copy a faer matrix into a row-major ndarray (layout-agnostic).
fn to_ndarray(a: MatRef<f64>) -> Array2<f64> {
    Array2::from_shape_fn((a.nrows(), a.ncols()), |(i, j)| a[(i, j)])
}

/// Build a faer matrix from a row-major ndarray.
fn from_ndarray(a: &ArrayView2<f64>) -> Mat<f64> {
    let (rows, cols) = a.dim();
    Mat::from_fn(rows, cols, |i, j| {
        a.get((i.unbound(), j.unbound())).copied().unwrap()
    })
}
