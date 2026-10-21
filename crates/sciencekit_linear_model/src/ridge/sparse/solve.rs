//! Sparse ridge solver resolution and orchestration (spec `ridge-regression`).
//!
//! `Automatic` resolves to `SparseCg` for sparse input; `Cholesky` builds the
//! centered Gram from the sparse structure and solves it with the backend's
//! Cholesky factor; `Svd` is dense-only and rejects sparse input.

use ndarray::{Array1, Array2, ArrayView1};
use sciencekit_common::{SKBackendKind, SKError};
use sciencekit_math::SKBackendSelection;
use sprs::CsMatView;

use super::cg::solve_with_sparse_cg;
use crate::ridge::builder::SKRidgeSolver;
use crate::ridge::gram::SKCenteredGramState;
use crate::ridge::solving::{SKRidgeSolution, solve_with_cholesky};

/// Solve one sparse ridge fit, resolving the configured solver by
/// representation. `Svd` is dense-only: a sparse input returns a structured
/// representation error pointing at `Cholesky`/`SparseCg`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn solve_sparse_ridge<F: SKBackendSelection>(
    configured: SKRidgeSolver,
    gram: &SKCenteredGramState<F>,
    features: CsMatView<F>,
    target: ArrayView1<F>,
    include_intercept: bool,
    alphas: &[F],
    tolerance: F,
    maximum_number_of_iterations: usize,
    backend: SKBackendKind,
) -> Result<SKRidgeSolution<F>, SKError> {
    match configured {
        SKRidgeSolver::Svd => Err(SKError::UnsupportedRepresentation {
            representation: "csr",
            suggestion: "use the Cholesky or SparseCg solver for sparse input",
        }),
        SKRidgeSolver::Cholesky => solve_sparse_cholesky(gram, include_intercept, alphas, backend),
        SKRidgeSolver::Automatic | SKRidgeSolver::SparseCg => solve_with_sparse_cg(
            features,
            target,
            include_intercept,
            alphas,
            tolerance,
            maximum_number_of_iterations,
            backend,
        ),
    }
}

/// Solve `(Ĉ + αI)β = ĉ` from the sparse-accumulated centered Gram, recovering
/// the intercept as `β₀ = ȳ − x̄ᵀβ`.
fn solve_sparse_cholesky<F: SKBackendSelection>(
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
