//! The opt-in `blas-backend` backend backed by `ndarray-linalg` over the
//! configured `blas-src` library (OpenBLAS/BLIS/MKL/Accelerate).
//!
//! Gated behind the `blas-backend` feature, which is never enabled by default.
//! When enabled, heavy dense algebra resolves to this backend; the pure-Rust
//! [`SKFaerBackend`] remains available as a fallback.

use ndarray::{Array1, Array2, ArrayView2};
use ndarray_linalg::{
    Cholesky, Diag, Eigh, Inverse, LeastSquaresSvd, QR, SVD, Solve, SolveTriangular, UPLO,
    layout::{AllocatedArray, AllocatedArrayMut},
    types::Lapack,
};
use sciencekit_common::{SKBackendKind, SKError};

use super::decompositions::{
    SKLUDecomposition, SKLeastSquaresSolution, SKQRDecomposition, SKSingularValueDecomposition,
};
use super::kernel::SKMathBackend;

/// The system-BLAS backend over `ndarray-linalg` + `blas-src`.
#[derive(Debug, Clone, Copy, Default)]
pub struct SKNdArrayLinalgBackend;

impl SKNdArrayLinalgBackend {
    /// Create the backend.
    pub fn new() -> Self {
        SKNdArrayLinalgBackend
    }
}

/// Shared implementation of the generic [`SKMathBackend`] surface over `F`.
macro_rules! impl_math_backend {
    ($float:ty) => {
        impl SKMathBackend<$float> for SKNdArrayLinalgBackend {
            fn kind(&self) -> SKBackendKind {
                SKBackendKind::NdArrayLinalg
            }

            fn gemm(
                &self,
                a: ArrayView2<$float>,
                b: ArrayView2<$float>,
                alpha: $float,
                _parallelism: usize,
            ) -> Array2<$float> {
                a.dot(&b) * alpha
            }

            fn svd(
                &self,
                a: ArrayView2<$float>,
            ) -> Result<SKSingularValueDecomposition<$float>, SKError> {
                let (u, singular_values, vh) = a
                    .svd(true, true)
                    .map_err(|error| SKError::Conversion(error.to_string()))?;
                let u = u.ok_or_else(|| SKError::Conversion("SVD did not return U".into()))?;
                let vh = vh.ok_or_else(|| SKError::Conversion("SVD did not return V".into()))?;
                Ok(SKSingularValueDecomposition {
                    u,
                    singular_values: singular_values.to_vec(),
                    // ndarray-linalg returns `Vᵀ`; store `V` (faer convention:
                    // columns of `V` are the right singular vectors).
                    v: vh.reversed_axes(),
                })
            }

            fn qr(&self, a: ArrayView2<$float>) -> SKQRDecomposition<$float> {
                let (q, r) = a.qr().expect("ndarray-linalg QR returned an error");
                SKQRDecomposition { q, r }
            }

            fn cholesky(&self, a: ArrayView2<$float>) -> Result<Array2<$float>, SKError> {
                a.cholesky(UPLO::Lower)
                    .map_err(|error| SKError::Conversion(error.to_string()))
            }

            fn solve_triangular(
                &self,
                a: ArrayView2<$float>,
                b: ArrayView2<$float>,
                lower: bool,
                unit_diagonal: bool,
            ) -> Result<Array2<$float>, SKError> {
                let uplo = if lower { UPLO::Lower } else { UPLO::Upper };
                let diag = if unit_diagonal {
                    Diag::Unit
                } else {
                    Diag::NonUnit
                };
                a.solve_triangular(uplo, diag, &b.to_owned())
                    .map_err(|error| SKError::Conversion(error.to_string()))
            }

            fn solve(
                &self,
                a: ArrayView2<$float>,
                b: ArrayView2<$float>,
            ) -> Result<Array2<$float>, SKError> {
                let mut solution = Array2::zeros((a.ncols(), b.ncols()));
                for column in 0..b.ncols() {
                    let rhs = b.column(column);
                    let solved = a
                        .solve(&rhs)
                        .map_err(|error| SKError::Conversion(error.to_string()))?;
                    solution.column_mut(column).assign(&solved);
                }
                Ok(solution)
            }

            fn eigh(
                &self,
                a: ArrayView2<$float>,
            ) -> Result<(Array1<$float>, Array2<$float>), SKError> {
                let (eigenvalues, eigenvectors) = a
                    .eigh(UPLO::Lower)
                    .map_err(|error| SKError::Conversion(error.to_string()))?;
                Ok((eigenvalues, eigenvectors))
            }

            fn lu(&self, a: ArrayView2<$float>) -> Result<SKLUDecomposition<$float>, SKError> {
                let n = a.nrows();
                let mut a_owned = a.to_owned();
                let layout = a_owned
                    .layout()
                    .map_err(|error| SKError::Conversion(error.to_string()))?;
                let ipiv = <$float as Lapack>::lu(layout, a_owned.as_allocated_mut().unwrap())
                    .map_err(|error| SKError::Conversion(error.to_string()))?;
                let mut l = Array2::<$float>::eye(n);
                let mut u = Array2::<$float>::zeros((n, n));
                for i in 0..n {
                    for j in 0..n {
                        if i > j {
                            l[(i, j)] = a_owned[(i, j)];
                        } else {
                            u[(i, j)] = a_owned[(i, j)];
                        }
                    }
                }
                let mut pivot: Vec<usize> = (0..n).collect();
                for (index, &value) in ipiv.iter().enumerate() {
                    pivot.swap(index, (value - 1) as usize);
                }
                Ok(SKLUDecomposition { pivot, l, u })
            }

            fn inv(&self, a: ArrayView2<$float>) -> Result<Array2<$float>, SKError> {
                a.inv()
                    .map_err(|error| SKError::Conversion(error.to_string()))
            }

            fn lstsq(
                &self,
                a: ArrayView2<$float>,
                b: ArrayView2<$float>,
            ) -> Result<SKLeastSquaresSolution<$float>, SKError> {
                let result = a
                    .least_squares(&b)
                    .map_err(|error| SKError::Conversion(error.to_string()))?;
                Ok(SKLeastSquaresSolution {
                    solution: result.solution,
                    rank: result.rank as usize,
                    singular_values: result.singular_values.to_vec(),
                    residual_sum_of_squares: result
                        .residual_sum_of_squares
                        .map(|residuals| residuals.iter().copied().collect()),
                })
            }
        }
    };
}

impl_math_backend!(f64);
impl_math_backend!(f32);
