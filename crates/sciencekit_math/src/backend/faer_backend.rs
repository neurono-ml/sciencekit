//! The pure-Rust faer backend and the matrixmultiply GEMM fallback
//! (spec `blas-interface`).

use faer::linalg::matmul::matmul;
use faer::linalg::solvers::{DenseSolveCore, Solve};
use faer::{Accum, Mat, MatRef, Par, Side, Unbind};
use ndarray::{Array1, Array2, ArrayView, ArrayView2, Ix2};
use sciencekit_common::{SKBackendKind, SKError, SKFloat};

use super::decompositions::{
    SKLUDecomposition, SKLeastSquaresSolution, SKQRDecomposition, SKSingularValueDecomposition,
};
use super::kernel::SKMathBackend;

/// The pure-Rust default backend backed by `faer` (Decision 1).
///
/// GEMM uses faer's blocked `matmul`, honouring the execution plan's
/// `parallelism`; decompositions delegate to faer's high-level SVD/QR/
/// Cholesky/LU/eigendecomposition. No C/Fortran BLAS is linked.
#[derive(Debug, Clone, Copy, Default)]
pub struct SKFaerBackend;

impl SKFaerBackend {
    /// Create the backend.
    pub fn new() -> Self {
        SKFaerBackend
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

/// A zero-copy-or-owned wrapper over a faer matrix view.
///
/// Built from a zero-copy ndarray view when the input is C-contiguous
/// (row-major), otherwise from an owned contiguous copy (the copy is internal
/// to the backend, per the design's risk note).
enum FaerInput<'a, F: SKFloat> {
    Borrowed(MatRef<'a, F>),
    Owned(Mat<F>),
}

impl<F: SKFloat> FaerInput<'_, F> {
    /// Return the matrix view.
    fn as_ref(&self) -> MatRef<'_, F> {
        match self {
            FaerInput::Borrowed(view) => *view,
            FaerInput::Owned(owned) => owned.as_ref(),
        }
    }
}

/// Build a zero-copy faer view over a C-contiguous ndarray row-major input,
/// falling back to an owned copy for strided/non-standard layouts.
fn to_faer<'a, F: SKFloat>(a: ArrayView<'a, F, Ix2>) -> FaerInput<'a, F> {
    if let Some(slice) = a.as_slice() {
        // SAFETY: `ArrayView<'a, F, Ix2>` owns (borrows) its data for `'a`, so
        // the row-major slice produced by `as_slice()` also lives for `'a`.
        // This only widens a lifetime; no data is (re)interpreted.
        let slice: &'a [F] = unsafe { std::mem::transmute(slice) };
        return FaerInput::Borrowed(MatRef::from_row_major_slice(slice, a.nrows(), a.ncols()));
    }
    FaerInput::Owned(Mat::from_fn(a.nrows(), a.ncols(), |i, j| a[(i, j)]))
}

/// Copy a faer matrix into an owned row-major ndarray.
fn to_ndarray<F: SKFloat>(m: &Mat<F>) -> Array2<F> {
    Array2::from_shape_fn((m.nrows(), m.ncols()), |(i, j)| m[(i, j)])
}

/// Copy an ndarray view into a faer matrix (owned, contiguous).
fn to_ndarray_copy<F: SKFloat>(a: &ArrayView2<F>) -> Mat<F> {
    Mat::from_fn(a.nrows(), a.ncols(), |i, j| a[(i, j)])
}

/// Map a faer error into the crate's central error taxonomy.
fn to_sk_error(message: impl std::fmt::Debug) -> SKError {
    SKError::Conversion(format!("{message:?}"))
}

/// matrixmultiply GEMM (single-threaded; the fallback path is explicit and
/// does not claim plan-level parallelism).
fn matrixmultiply_gemm<F: SKFloat>(
    a: ArrayView2<F>,
    b: ArrayView2<F>,
    alpha: F,
    _parallelism: usize,
) -> Array2<F> {
    let (rows, inner, cols) = (a.nrows(), a.ncols(), b.ncols());
    let mut output = Array2::<F>::zeros((rows, cols));
    // SAFETY: matrixmultiply only reads `a`/`b` and writes `output` within
    // their declared bounds; the strides describe the actual ndarray layout.
    unsafe {
        if std::any::TypeId::of::<F>() == std::any::TypeId::of::<f64>() {
            matrixmultiply::dgemm(
                rows,
                inner,
                cols,
                alpha.to_f64().unwrap_or(1.0),
                a.as_ptr() as *const f64,
                a.strides()[0],
                a.strides()[1],
                b.as_ptr() as *const f64,
                b.strides()[0],
                b.strides()[1],
                0.0,
                output.as_mut_ptr() as *mut f64,
                output.strides()[0],
                output.strides()[1],
            );
        } else {
            matrixmultiply::sgemm(
                rows,
                inner,
                cols,
                alpha.to_f32().unwrap_or(1.0),
                a.as_ptr() as *const f32,
                a.strides()[0],
                a.strides()[1],
                b.as_ptr() as *const f32,
                b.strides()[0],
                b.strides()[1],
                0.0,
                output.as_mut_ptr() as *mut f32,
                output.strides()[0],
                output.strides()[1],
            );
        }
    }
    output
}

/// faer GEMM honouring the plan's parallelism for a concrete float type.
macro_rules! faer_gemm_for {
    ($name:ident, $float:ty) => {
        fn $name(
            a: ArrayView2<$float>,
            b: ArrayView2<$float>,
            alpha: $float,
            parallelism: usize,
        ) -> Array2<$float> {
            let a = to_faer(a);
            let b = to_faer(b);
            let rows = a.as_ref().nrows();
            let cols = b.as_ref().ncols();
            let mut output = Mat::<$float>::zeros(rows, cols);
            let par = if parallelism > 1 { Par::rayon(parallelism) } else { Par::Seq };
            matmul(
                output.as_mut(),
                Accum::Replace,
                a.as_ref(),
                b.as_ref(),
                alpha,
                par,
            );
            to_ndarray(&output)
        }
    };
}

faer_gemm_for!(faer_gemm_f64, f64);
faer_gemm_for!(faer_gemm_f32, f32);

/// Shared implementation of the generic [`SKMathBackend`] surface over a
/// concrete float type (the sealed [`SKFloat`] bound admits exactly `f32` and
/// `f64`).
macro_rules! impl_math_backend {
    ($float:ty, $backend:ty, $kind:expr, $gemm:ident) => {
        impl SKMathBackend<$float> for $backend {
            fn kind(&self) -> SKBackendKind {
                $kind
            }

            fn gemm(
                &self,
                a: ArrayView2<$float>,
                b: ArrayView2<$float>,
                alpha: $float,
                parallelism: usize,
            ) -> Array2<$float> {
                $gemm(a, b, alpha, parallelism)
            }

            fn svd(
                &self,
                a: ArrayView2<$float>,
            ) -> Result<SKSingularValueDecomposition<$float>, SKError> {
                let a = to_faer(a);
                let decomposition = a.as_ref().to_owned().svd().map_err(to_sk_error)?;
                let mut singular_values = Vec::new();
                decomposition
                    .S()
                    .for_each(|&value| singular_values.push(value));
                Ok(SKSingularValueDecomposition {
                    u: to_ndarray(&decomposition.U().to_owned()),
                    singular_values,
                    v: to_ndarray(&decomposition.V().to_owned()),
                })
            }

            fn qr(&self, a: ArrayView2<$float>) -> SKQRDecomposition<$float> {
                let a = to_faer(a);
                let decomposition = a.as_ref().to_owned().qr();
                SKQRDecomposition {
                    q: to_ndarray(&decomposition.compute_Q()),
                    r: to_ndarray(&decomposition.R().to_owned()),
                }
            }

            fn cholesky(&self, a: ArrayView2<$float>) -> Result<Array2<$float>, SKError> {
                let a = to_faer(a);
                let factor = a.as_ref().to_owned().llt(Side::Lower).map_err(to_sk_error)?;
                Ok(to_ndarray(&factor.L().to_owned()))
            }

            fn solve_triangular(
                &self,
                a: ArrayView2<$float>,
                b: ArrayView2<$float>,
                lower: bool,
                unit_diagonal: bool,
            ) -> Result<Array2<$float>, SKError> {
                if a.nrows() != b.nrows() {
                    return Err(SKError::shape_mismatch_2d(
                        a.nrows(),
                        a.ncols(),
                        b.nrows(),
                        b.ncols(),
                    ));
                }
                let a = to_faer(a);
                let mut rhs = to_ndarray_copy(&b);
                match (lower, unit_diagonal) {
                    (true, false) => a.as_ref().solve_lower_triangular_in_place(rhs.as_mut()),
                    (true, true) => a.as_ref().solve_unit_lower_triangular_in_place(rhs.as_mut()),
                    (false, false) => a.as_ref().solve_upper_triangular_in_place(rhs.as_mut()),
                    (false, true) => a.as_ref().solve_unit_upper_triangular_in_place(rhs.as_mut()),
                }
                Ok(to_ndarray(&rhs))
            }

            fn solve(
                &self,
                a: ArrayView2<$float>,
                b: ArrayView2<$float>,
            ) -> Result<Array2<$float>, SKError> {
                if a.nrows() != a.ncols() {
                    return Err(SKError::shape_mismatch_2d(
                        a.nrows(),
                        a.ncols(),
                        a.nrows(),
                        a.ncols(),
                    ));
                }
                let a = to_faer(a);
                let decomposition = a.as_ref().to_owned().partial_piv_lu();
                let b_mat = to_ndarray_copy(&b);
                let solution = decomposition.solve(b_mat.as_ref());
                Ok(to_ndarray(&solution))
            }

            fn eigh(
                &self,
                a: ArrayView2<$float>,
            ) -> Result<(Array1<$float>, Array2<$float>), SKError> {
                let a = to_faer(a);
                let decomposition = a
                    .as_ref()
                    .to_owned()
                    .self_adjoint_eigen(Side::Lower)
                    .map_err(to_sk_error)?;
                let mut eigenvalues = Vec::new();
                decomposition
                    .S()
                    .for_each(|&value| eigenvalues.push(value));
                Ok((
                    Array1::from(eigenvalues),
                    to_ndarray(&decomposition.U().to_owned()),
                ))
            }

            fn lu(&self, a: ArrayView2<$float>) -> Result<SKLUDecomposition<$float>, SKError> {
                let a = to_faer(a);
                let decomposition = a.as_ref().to_owned().partial_piv_lu();
                let (forward, _) = decomposition.P().arrays();
                let pivot = forward.iter().map(|index| index.unbound()).collect();
                Ok(SKLUDecomposition {
                    pivot,
                    l: to_ndarray(&decomposition.L().to_owned()),
                    u: to_ndarray(&decomposition.U().to_owned()),
                })
            }

            fn inv(&self, a: ArrayView2<$float>) -> Result<Array2<$float>, SKError> {
                let n = a.nrows();
                if a.nrows() != a.ncols() {
                    return Err(SKError::shape_mismatch_2d(n, n, a.nrows(), a.ncols()));
                }
                let a = to_faer(a);
                let decomposition = a.as_ref().to_owned().partial_piv_lu();
                Ok(to_ndarray(&decomposition.inverse()))
            }

            fn lstsq(
                &self,
                a: ArrayView2<$float>,
                b: ArrayView2<$float>,
            ) -> Result<SKLeastSquaresSolution<$float>, SKError> {
                let a_faer = to_faer(a);
                let decomposition = a_faer.as_ref().to_owned().svd().map_err(to_sk_error)?;
                let b_mat = to_ndarray_copy(&b);
                // Minimum-norm least-squares solution `x = pinv(A) b` via the
                // SVD pseudoinverse (Decision 7: faer composes from SVD).
                let solution = decomposition.pseudoinverse() * b_mat.as_ref();
                let mut singular_values = Vec::new();
                decomposition
                    .S()
                    .for_each(|&value| singular_values.push(value));
                let cutoff = singular_values
                    .first()
                    .copied()
                    .unwrap_or(0.0)
                    * <$float>::EPSILON
                    * (a.nrows().max(a.ncols()) as $float);
                let rank = singular_values
                    .iter()
                    .filter(|&&value| value > cutoff)
                    .count();
                let solution = to_ndarray(&solution);
                let residual_sum_of_squares = if a.nrows() > a.ncols() {
                    let ax = a.dot(&solution);
                    let difference = &b - &ax;
                    Some(
                        difference
                            .axis_iter(ndarray::Axis(1))
                            .map(|column| column.mapv(|value| value * value).sum())
                            .collect(),
                    )
                } else {
                    None
                };
                Ok(SKLeastSquaresSolution {
                    solution,
                    rank,
                    singular_values,
                    residual_sum_of_squares,
                })
            }
        }
    };
}

impl_math_backend!(f64, SKFaerBackend, SKBackendKind::Faer, faer_gemm_f64);
impl_math_backend!(f32, SKFaerBackend, SKBackendKind::Faer, faer_gemm_f32);
impl_math_backend!(
    f64,
    SKMatrixMultiplyBackend,
    SKBackendKind::MatrixMultiply,
    matrixmultiply_gemm
);
impl_math_backend!(
    f32,
    SKMatrixMultiplyBackend,
    SKBackendKind::MatrixMultiply,
    matrixmultiply_gemm
);