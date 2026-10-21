//! The coordinate-descent solver result types (spec
//! `coordinate-descent-solver`).

use ndarray::{Array1, Array2};

/// One `alpha` solve's result: the coefficient vector, the intercept, the
/// sweeps spent and how many coordinates screening excluded.
#[derive(Debug)]
pub(crate) struct SKCoordinateDescentAlphaSolution<F> {
    /// The learned coefficients, one per design column.
    pub(crate) coefficients: Array1<F>,
    /// The unpenalized intercept.
    pub(crate) intercept: F,
    /// The total sweeps spent (across reactivations).
    pub(crate) iterations: usize,
    /// How many coordinates were excluded from the active set.
    pub(crate) screened_count: usize,
}

/// Every `alpha` solve's aggregate result: the `n_alphas × n_features`
/// coefficients, the per-alpha intercepts and the effort/screening totals.
#[derive(Debug)]
pub(crate) struct SKCoordinateDescentSolution<F> {
    /// The learned coefficients, shaped `n_alphas × n_features`.
    pub(crate) coefficients: Array2<F>,
    /// The per-alpha intercepts.
    pub(crate) intercepts: Array1<F>,
    /// The total sweeps spent across the alphas.
    pub(crate) iterations: usize,
    /// The total coordinates excluded across the alphas.
    pub(crate) screened_count: usize,
}
