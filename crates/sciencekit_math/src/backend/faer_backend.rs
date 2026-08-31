//! The pure-Rust faer backend and the matrixmultiply GEMM fallback
//! (spec `blas-interface`).

use faer::linalg::matmul::matmul;
use faer::{Accum, Mat, MatRef, Par};
use sciencekit_common::SKError;

use super::{SKMathBackend, SKQRDecomposition, SKSingularValueDecomposition};

/// The pure-Rust default backend backed by `faer` (Decision 1).
///
/// GEMM uses faer's blocked `matmul`; decompositions delegate to faer's
/// high-level SVD/QR/Cholesky. No C/Fortran BLAS is linked.
#[derive(Debug, Clone, Copy, Default)]
pub struct SKFaerBackend;

impl SKFaerBackend {
    /// Create the backend.
    pub fn new() -> Self {
        SKFaerBackend
    }
}

impl SKMathBackend for SKFaerBackend {
    fn gemm(&self, a: MatRef<f64>, b: MatRef<f64>, alpha: f64) -> Mat<f64> {
        let rows = a.nrows();
        let cols = b.ncols();
        let mut output = Mat::<f64>::zeros(rows, cols);
        matmul(output.as_mut(), Accum::Replace, a, b, alpha, Par::Seq);
        output
    }

    fn svd(&self, a: MatRef<f64>) -> Result<SKSingularValueDecomposition, SKError> {
        let decomposition = a
            .to_owned()
            .svd()
            .map_err(|error| SKError::Conversion(format!("{error:?}")))?;
        let mut singular_values = Vec::new();
        decomposition
            .S()
            .for_each(|&value| singular_values.push(value));
        Ok(SKSingularValueDecomposition {
            u: decomposition.U().to_owned(),
            singular_values,
            v: decomposition.V().to_owned(),
        })
    }

    fn qr(&self, a: MatRef<f64>) -> SKQRDecomposition {
        let decomposition = a.to_owned().qr();
        SKQRDecomposition {
            q: decomposition.compute_Q(),
            r: decomposition.R().to_owned(),
        }
    }

    fn cholesky(&self, a: MatRef<f64>) -> Result<Mat<f64>, SKError> {
        let factor = a
            .to_owned()
            .llt(faer::Side::Lower)
            .map_err(|error| SKError::Conversion(format!("{error:?}")))?;
        Ok(factor.L().to_owned())
    }
}

/// A GEMM-fallback backend using `matrixmultiply`'s dense kernel.
///
/// Heavy GEMM is dispatched to `matrixmultiply` (pure Rust); the
/// decompositions are not provided by that crate, so they delegate to faer,
/// which is always available. Useful as an explicit, non-faer GEMM path.
#[derive(Debug, Clone, Copy, Default)]
pub struct SKMatrixMultiplyBackend;

impl SKMatrixMultiplyBackend {
    /// Create the backend.
    pub fn new() -> Self {
        SKMatrixMultiplyBackend
    }
}

impl SKMathBackend for SKMatrixMultiplyBackend {
    fn gemm(&self, a: MatRef<f64>, b: MatRef<f64>, alpha: f64) -> Mat<f64> {
        let (rows, inner, cols) = (a.nrows(), a.ncols(), b.ncols());
        let mut output = Mat::<f64>::zeros(rows, cols);
        // SAFETY: matrixmultiply only reads `a`/`b` and writes `output` within
        // their declared bounds; the strides describe the actual faer layout.
        unsafe {
            matrixmultiply::dgemm(
                rows,
                inner,
                cols,
                alpha,
                a.as_ptr(),
                a.row_stride(),
                a.col_stride(),
                b.as_ptr(),
                b.row_stride(),
                b.col_stride(),
                0.0,
                output.as_mut().as_ptr_mut(),
                output.row_stride(),
                output.col_stride(),
            );
        }
        output
    }

    fn svd(&self, a: MatRef<f64>) -> Result<SKSingularValueDecomposition, SKError> {
        SKFaerBackend.svd(a)
    }

    fn qr(&self, a: MatRef<f64>) -> SKQRDecomposition {
        SKFaerBackend.qr(a)
    }

    fn cholesky(&self, a: MatRef<f64>) -> Result<Mat<f64>, SKError> {
        SKFaerBackend.cholesky(a)
    }
}
