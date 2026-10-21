//! The data types shared across the ridge solve modules (spec
//! `ridge-regression`).

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

use crate::ridge::builder::SKRidgeSolver;

/// The cross-product inputs a dense ridge solve consumes, already in the form
/// the selected solver expects: centered when the intercept is included, raw
/// otherwise.
pub(super) struct SKRidgeSolveInputs<'a, F> {
    /// `Ĉ = XᵀX − n·x̄x̄ᵀ` (intercept) or the raw `XᵀX`.
    pub gram: ArrayView2<'a, F>,
    /// `ĉ = Xᵀy − n·x̄·ȳ` (intercept) or the raw `Xᵀy`.
    pub cross: ArrayView1<'a, F>,
    /// The centered design (intercept) or the raw design, for the SVD path.
    pub design: ArrayView2<'a, F>,
    /// The centered target (intercept) or the raw target, for the SVD path.
    pub target: ArrayView1<'a, F>,
    /// The penalty values, in configuration order.
    pub alphas: &'a [F],
}

/// The coefficients, rank and singular values one dense solve produced.
pub(super) struct SKRidgeSolveResult<F> {
    pub(super) coefficients: Array2<F>,
    pub(super) rank: usize,
    pub(super) singular_values: Vec<F>,
}

/// One dense ridge solve's complete result: the resolved solver, the
/// `n_alphas × n_features` coefficients, the per-alpha intercepts, the reported
/// rank and the singular values.
pub(crate) struct SKRidgeSolution<F> {
    /// The solver actually used after resolution and any fallback.
    pub resolved: SKRidgeSolver,
    /// The learned coefficients, shaped `n_alphas × n_features`.
    pub coefficients: Array2<F>,
    /// The per-alpha intercepts.
    pub intercepts: Array1<F>,
    /// The reported rank of the factored matrix.
    pub rank: usize,
    /// The singular values (empty when the Cholesky path ran).
    pub singular_values: Vec<F>,
}
