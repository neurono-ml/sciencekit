//! The logistic solver result type (spec `logistic-solver`).

use ndarray::{Array1, Array2};

/// A solved logistic model: the per-head coefficient rows and the per-head
/// intercepts.
#[derive(Debug, Clone)]
pub(crate) struct SKLogisticSolution<F> {
    /// The coefficient rows, shaped `heads × features` (one head for a binary
    /// fit).
    pub(crate) coefficients: Array2<F>,
    /// The per-head intercepts.
    pub(crate) intercepts: Array1<F>,
}
