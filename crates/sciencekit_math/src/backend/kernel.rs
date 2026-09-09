//! The `SKMathBackend` abstraction and the norm-order enum (spec
//! `blas-interface`).
//!
//! The trait is the single host-centric interface for heavy dense algebra: it
//! is generic over the sealed [`SKFloat`] scalar bound, accepts zero-copy
//! ndarray views and returns owned ndarray results. No concrete backend type
//! (`faer`, `ndarray-linalg`) appears in this surface.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use sciencekit_common::{SKBackendKind, SKError, SKFloat};

use super::decompositions::{
    SKLUDecomposition, SKLeastSquaresSolution, SKQRDecomposition, SKSingularValueDecomposition,
};

/// The dense linear-algebra backend abstraction.
///
/// Implementations are pure and deterministic and must be `Send + Sync` so an
/// execution plan can dispatch heavy algebra across threads. The surface is
/// host-centric: inputs are zero-copy [`ArrayView2`] views and results are
/// owned [`Array2`] containers, independent of the concrete backend.
pub trait SKMathBackend<F: SKFloat>: Send + Sync {
    /// The backend kind, recorded in observability spans.
    fn kind(&self) -> SKBackendKind;

    /// Compute `C = α · A·B` for dense `A (m×k)` and `B (k×n)`.
    ///
    /// `parallelism` is the execution plan's resolved parallelism level; the
    /// backend honours it for internally parallelizable products.
    fn gemm(&self, a: ArrayView2<F>, b: ArrayView2<F>, alpha: F, parallelism: usize) -> Array2<F>;

    /// Compute the singular value decomposition of `A`.
    fn svd(&self, a: ArrayView2<F>) -> Result<SKSingularValueDecomposition<F>, SKError>;

    /// Compute the (thin) QR decomposition of `A`.
    fn qr(&self, a: ArrayView2<F>) -> SKQRDecomposition<F>;

    /// Compute the Cholesky factor `L` of a positive-definite `A` (`A = L Lᵀ`).
    fn cholesky(&self, a: ArrayView2<F>) -> Result<Array2<F>, SKError>;

    /// Solve a triangular system `A x = b` (`A` upper or lower triangular).
    fn solve_triangular(
        &self,
        a: ArrayView2<F>,
        b: ArrayView2<F>,
        lower: bool,
        unit_diagonal: bool,
    ) -> Result<Array2<F>, SKError>;

    /// Solve the general square system `A x = b`.
    fn solve(&self, a: ArrayView2<F>, b: ArrayView2<F>) -> Result<Array2<F>, SKError>;

    /// Symmetric eigendecomposition of `A`, returning eigenvalues (ascending)
    /// and eigenvectors as columns.
    fn eigh(&self, a: ArrayView2<F>) -> Result<(Array1<F>, Array2<F>), SKError>;

    /// Compute the LU decomposition `P A = L U` with a host-normalized row
    /// permutation.
    fn lu(&self, a: ArrayView2<F>) -> Result<SKLUDecomposition<F>, SKError>;

    /// Log-determinant of `A`, returned as `(sign, log_abs_det)` without
    /// materializing the determinant (no underflow).
    ///
    /// Default implementation: derives `sign` and `log_abs_det` from the LU
    /// factorization (Decision 3), never materializing `det`.
    fn slogdet(&self, a: ArrayView2<F>) -> Result<(F, F), SKError> {
        let decomposition = self.lu(a)?;
        let n = a.nrows();
        if a.nrows() != a.ncols() {
            return Err(SKError::shape_mismatch_2d(n, n, a.nrows(), a.ncols()));
        }
        // sign(det) = sign(permutation) * ∏ sign(U[i,i])
        // log_abs_det = ∑ ln|U[i,i]|
        let permutation_sign: F = sk_permutation_sign(&decomposition.pivot);
        let mut sign: F = permutation_sign;
        let mut log_abs_det: F = F::zero();
        for i in 0..n {
            let diagonal = decomposition.u[(i, i)];
            if diagonal == F::zero() {
                return Ok((F::zero(), F::neg_infinity()));
            }
            if diagonal < F::zero() {
                sign = -sign;
            }
            log_abs_det = log_abs_det + diagonal.abs().ln();
        }
        Ok((sign, log_abs_det))
    }

    /// Moore–Penrose pseudoinverse of `A`, computed via the SVD.
    ///
    /// Default implementation: `pinv = V Σ⁺ Uᵀ` where `Σ⁺` inverts the
    /// singular values above `rcond · σ_max` (`rcond = F::epsilon()`) and
    /// zeroes the rest, matching numpy/scipy defaults.
    fn pinv(&self, a: ArrayView2<F>) -> Result<Array2<F>, SKError> {
        let (m, n) = a.dim();
        let svd = self.svd(a)?;
        let cutoff = svd.singular_values.first().copied().unwrap_or(F::zero())
            * F::epsilon()
            * F::from(m.max(n)).unwrap();
        let mut sigma_plus = Array2::<F>::zeros((n, m));
        for (index, &value) in svd.singular_values.iter().enumerate() {
            if value > cutoff {
                sigma_plus[(index, index)] = F::one() / value;
            }
        }
        Ok(svd.v.dot(&sigma_plus).dot(&svd.u.t()))
    }

    /// Inverse of a square matrix `A`.
    ///
    /// Default implementation: solves `A x = I` via [`SKMathBackend::solve`].
    fn inv(&self, a: ArrayView2<F>) -> Result<Array2<F>, SKError> {
        let n = a.nrows();
        if a.nrows() != a.ncols() {
            return Err(SKError::shape_mismatch_2d(n, n, a.nrows(), a.ncols()));
        }
        let identity = Array2::<F>::eye(n);
        self.solve(a, identity.view())
    }

    /// Minimum-norm least-squares solution `x = argmin ‖b - A x‖₂`.
    ///
    /// Default implementation: composes `x = pinv(A) b` from the SVD (see
    /// Decision 7); backends with a native driver (ndarray-linalg `gelsd`)
    /// may override it.
    fn lstsq(
        &self,
        a: ArrayView2<F>,
        b: ArrayView2<F>,
    ) -> Result<SKLeastSquaresSolution<F>, SKError> {
        let svd = self.svd(a)?;
        let cutoff = svd.singular_values.first().copied().unwrap_or(F::zero())
            * F::epsilon()
            * F::from(a.nrows().max(a.ncols())).unwrap();
        let rank = svd
            .singular_values
            .iter()
            .filter(|&&value| value > cutoff)
            .count();
        let solution = self.pinv(a)?.dot(&b);
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
            singular_values: svd.singular_values,
            residual_sum_of_squares,
        })
    }

    /// Compute a matrix norm of `A` for the given order.
    ///
    /// Default implementation: host-side reductions, with the spectral,
    /// negative-spectral and nuclear orders routed through
    /// [`SKMathBackend::svd`]. `General` is only valid for vectors.
    fn norm(&self, a: ArrayView2<F>, ord: SKNormKind<F>) -> Result<F, SKError> {
        match ord {
            SKNormKind::Frobenius => Ok(a.mapv(|value| value * value).sum().sqrt()),
            SKNormKind::L1 => {
                let maximum = (0..a.ncols())
                    .map(|column| a.column(column).mapv(F::abs).sum())
                    .fold(F::neg_infinity(), F::max);
                Ok(if a.ncols() == 0 { F::zero() } else { maximum })
            }
            SKNormKind::Infinity => {
                let maximum = (0..a.nrows())
                    .map(|row| a.row(row).mapv(F::abs).sum())
                    .fold(F::neg_infinity(), F::max);
                Ok(if a.nrows() == 0 { F::zero() } else { maximum })
            }
            SKNormKind::NegativeInfinity => {
                let minimum = (0..a.nrows())
                    .map(|row| a.row(row).mapv(F::abs).sum())
                    .fold(F::infinity(), F::min);
                Ok(minimum)
            }
            SKNormKind::NegativeL1 => {
                let minimum = (0..a.ncols())
                    .map(|column| a.column(column).mapv(F::abs).sum())
                    .fold(F::infinity(), F::min);
                Ok(minimum)
            }
            SKNormKind::L2 => {
                let svd = self.svd(a)?;
                svd.singular_values
                    .first()
                    .copied()
                    .ok_or_else(|| SKError::Conversion("empty matrix norm".into()))
            }
            SKNormKind::NegativeL2 => {
                let svd = self.svd(a)?;
                svd.singular_values
                    .last()
                    .copied()
                    .ok_or_else(|| SKError::Conversion("empty matrix norm".into()))
            }
            SKNormKind::Nuclear => {
                let svd = self.svd(a)?;
                Ok(svd
                    .singular_values
                    .iter()
                    .copied()
                    .fold(F::zero(), |acc, value| acc + value))
            }
            SKNormKind::General { .. } => Err(SKError::Conversion(
                "General p-norms are only supported for vectors".into(),
            )),
        }
    }

    /// Compute a vector norm of `x` for the given order.
    ///
    /// Default implementation: host-side reductions; `General { order }`
    /// covers arbitrary p-norms `(∑|x|ᵖ)^(1/p)` (p = 0 counts non-zeros).
    fn vector_norm(&self, a: ArrayView1<F>, ord: SKNormKind<F>) -> Result<F, SKError> {
        match ord {
            SKNormKind::Frobenius | SKNormKind::L2 => {
                Ok(a.mapv(|value| value * value).sum().sqrt())
            }
            SKNormKind::L1 => Ok(a.mapv(F::abs).sum()),
            SKNormKind::Infinity => Ok(a.iter().copied().fold(F::neg_infinity(), F::max)),
            SKNormKind::NegativeInfinity => Ok(a.iter().copied().fold(F::infinity(), F::min)),
            SKNormKind::NegativeL1 => {
                let sum = a.mapv(|value| value.abs().recip()).sum();
                Ok(sum.recip())
            }
            SKNormKind::NegativeL2 => {
                let sum = a.mapv(|value| (value.abs().recip()).powi(2)).sum();
                Ok(sum.sqrt().recip())
            }
            SKNormKind::General { order } => {
                if order == F::zero() {
                    let count = a.iter().filter(|&&value| value != F::zero()).count();
                    Ok(
                        F::from(count)
                            .ok_or_else(|| SKError::Conversion("norm overflow".into()))?,
                    )
                } else {
                    let sum = a.mapv(|value| value.abs().powf(order)).sum();
                    Ok(sum.powf(F::one() / order))
                }
            }
            SKNormKind::Nuclear => Err(SKError::Conversion(
                "nuclear norm is only defined for matrices".into(),
            )),
        }
    }
}

/// Sign of the permutation encoded as a forward pivot array (`pivot[i]` =
/// original row at position `i`): `+1` for an even permutation, `-1` odd.
fn sk_permutation_sign<F: SKFloat>(pivot: &[usize]) -> F {
    let n = pivot.len();
    let mut parity = false;
    let mut visited = vec![false; n];
    for start in 0..n {
        if visited[start] {
            continue;
        }
        let mut length = 0usize;
        let mut cursor = start;
        while !visited[cursor] {
            visited[cursor] = true;
            length += 1;
            cursor = pivot[cursor];
        }
        // each cycle of length L contributes L-1 transpositions.
        if length % 2 == 0 {
            parity = !parity;
        }
    }
    if parity { -F::one() } else { F::one() }
}

/// The norm orders supported by [`SKMathBackend::norm`] and
/// [`SKMathBackend::vector_norm`].
///
/// Mirrors `numpy.linalg.norm`/`scipy.linalg.norm`: specialized paths for the
/// common matrix/vector orders plus a `General` p-norm fallback for arbitrary
/// orders.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SKNormKind<F: SKFloat> {
    /// Frobenius norm (matrix) / Euclidean norm (vector).
    Frobenius,
    /// `L2` norm: spectral (matrix) / Euclidean (vector).
    L2,
    /// `L1` norm: max column sum (matrix) / sum of absolutes (vector).
    L1,
    /// Infinity norm: max row sum (matrix) / max absolute (vector).
    Infinity,
    /// Negative infinity norm: min row sum (matrix) / min absolute (vector).
    NegativeInfinity,
    /// Negative `L1`: min column sum (matrix).
    NegativeL1,
    /// Negative `L2`: smallest singular value (matrix).
    NegativeL2,
    /// Nuclear norm (matrix): sum of singular values.
    Nuclear,
    /// General/arbitrary p-norm fallback: `(∑|x|ᵖ)^(1/p)`.
    General {
        /// The order `p`.
        order: F,
    },
}
