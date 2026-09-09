//! Concrete, host-centric decomposition result containers (spec
//! `blas-interface`).
//!
//! Every result is an owned ndarray container generic over the sealed
//! [`SKFloat`] bound — independent of the backend that produced it. The LU
//! pivot is a host-normalized `Vec<usize>` row permutation (`P A = L U`).

use ndarray::Array2;
use sciencekit_common::SKFloat;

/// The result of a singular value decomposition `A = U Σ Vᵀ`.
#[derive(Debug, Clone)]
pub struct SKSingularValueDecomposition<F: SKFloat> {
    /// The left singular vectors `U`.
    pub u: Array2<F>,
    /// The singular values, non-increasing.
    pub singular_values: Vec<F>,
    /// The right singular vectors `V` (columns are `Vᵀ` rows).
    pub v: Array2<F>,
}

/// The result of a thin QR decomposition `A = Q R`.
#[derive(Debug, Clone)]
pub struct SKQRDecomposition<F: SKFloat> {
    /// The orthogonal factor `Q`.
    pub q: Array2<F>,
    /// The upper-triangular factor `R`.
    pub r: Array2<F>,
}

/// The result of an LU decomposition `P A = L U`.
#[derive(Debug, Clone)]
pub struct SKLUDecomposition<F: SKFloat> {
    /// The host-normalized row permutation: `pivot[i]` is the original row
    /// that ends up at position `i` (`P[i, pivot[i]] = 1`).
    pub pivot: Vec<usize>,
    /// The unit-lower-triangular factor `L`.
    pub l: Array2<F>,
    /// The upper-triangular factor `U`.
    pub u: Array2<F>,
}

/// The result of a minimum-norm least-squares solve (`A x = b`).
#[derive(Debug, Clone)]
pub struct SKLeastSquaresSolution<F: SKFloat> {
    /// The minimum-norm least-squares solution `x`.
    pub solution: Array2<F>,
    /// The effective rank of `A` after `rcond` truncation.
    pub rank: usize,
    /// The singular values of `A`, non-increasing.
    pub singular_values: Vec<F>,
    /// The squared residual norms per column of `b` (`‖b - A x‖₂²`), present
    /// for overdetermined systems (`rows(a) > cols(a)`).
    pub residual_sum_of_squares: Option<Vec<F>>,
}