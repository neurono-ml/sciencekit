//! The shared Gram path: solve the accumulated centered normal equations
//! without a dense design (spec `ridge-regression`).
//!
//! Serves the sparse-Cholesky, streaming and incremental regimes, which all
//! carry an [`SKCenteredGramState`] rather than a design matrix.

use ndarray::{Array1, Array2};
use sciencekit_common::{SKBackendKind, SKError};
use sciencekit_math::SKBackendSelection;

use super::cholesky::solve_with_cholesky;
use super::solve_types::SKRidgeSolution;
use crate::ridge::builder::SKRidgeSolver;
use crate::ridge::gram::SKCenteredGramState;

/// Solve `(Ĉ + αI)β = ĉ` once per `alpha` from the accumulated centered Gram,
/// recovering the intercept as `β₀ = ȳ − x̄ᵀβ`.
///
/// The resolved solver is always `Cholesky` (the Gram path), because the design
/// itself is no longer available for an SVD. Callers that support the
/// Cholesky-to-SVD fallback resolve it before calling.
pub(crate) fn solve_from_gram<F: SKBackendSelection>(
    gram: &SKCenteredGramState<F>,
    include_intercept: bool,
    alphas: &[F],
    backend: SKBackendKind,
) -> Result<SKRidgeSolution<F>, SKError> {
    let (centered_gram, centered_cross, feature_means, target_mean) = gram.centered();
    let (solve_gram, solve_cross) = if include_intercept {
        (centered_gram, centered_cross)
    } else {
        (gram.gram.clone(), gram.cross.clone())
    };
    let features_count = solve_gram.ncols();
    let mut coefficients = Array2::zeros((alphas.len(), features_count));
    for (row_index, &alpha) in alphas.iter().enumerate() {
        let mut system = solve_gram.clone();
        system.diag_mut().mapv_inplace(|value| value + alpha);
        let beta = solve_with_cholesky(backend, system.view(), solve_cross.view())?;
        coefficients.row_mut(row_index).assign(&beta);
    }
    let intercepts = if include_intercept {
        coefficients
            .outer_iter()
            .map(|row| target_mean - row.dot(&feature_means))
            .collect()
    } else {
        Array1::zeros(alphas.len())
    };
    Ok(SKRidgeSolution {
        resolved: SKRidgeSolver::Cholesky,
        coefficients,
        intercepts,
        rank: features_count,
        singular_values: Vec::new(),
    })
}
