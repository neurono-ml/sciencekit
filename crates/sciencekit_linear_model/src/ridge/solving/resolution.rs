//! Solver resolution and the dense solve orchestration: centering, solver
//! selection, per-alpha solves and intercept recovery
//! (spec `ridge-regression`).

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use sciencekit_common::{SKBackendKind, SKError};
use sciencekit_math::SKBackendSelection;

use super::centering::{center_design, center_target};
use super::cholesky::solve_with_cholesky;
use super::solve_types::{SKRidgeSolution, SKRidgeSolveInputs, SKRidgeSolveResult};
use super::svd::solve_with_svd;
use crate::ridge::builder::SKRidgeSolver;
use crate::ridge::gram::SKCenteredGramState;

/// Reject a sparse input with a structured representation error.
///
/// The `Svd` solver points the consumer at the sparse-capable solvers; until the
/// sparse paths land, every solver points at the dense conversion.
pub(crate) fn reject_sparse(configured: SKRidgeSolver) -> SKError {
    match configured {
        SKRidgeSolver::Svd => SKError::UnsupportedRepresentation {
            representation: "csr",
            suggestion: "use the Cholesky or SparseCg solver for sparse input",
        },
        _ => SKError::UnsupportedRepresentation {
            representation: "csr",
            suggestion: "sparse ridge fits arrive with the SparseCg solver; convert to dense with to_dense for now",
        },
    }
}

/// Resolve the configured solver for the input representation.
///
/// `Automatic` maps dense → `Cholesky` and sparse → `SparseCg`; every explicit
/// selection is honoured verbatim.
pub(super) fn resolve_solver(
    configured: SKRidgeSolver,
    representation: &'static str,
) -> SKRidgeSolver {
    match configured {
        SKRidgeSolver::Automatic => {
            if representation == "dense" {
                SKRidgeSolver::Cholesky
            } else {
                SKRidgeSolver::SparseCg
            }
        }
        explicit => explicit,
    }
}

/// Solve one ridge fit on a dense design, returning the resolved solver, the
/// `n_alphas × n_features` coefficients, the per-alpha intercepts, the rank and
/// the singular values.
///
/// When the intercept is included the inputs are centered and the intercept is
/// recovered as `β₀ = ȳ − x̄ᵀβ`; otherwise the raw Gram is solved and every
/// intercept is zero. `Cholesky` falls back to `Svd` when every `alpha` is zero
/// or the factor fails, and the fallback is what the model records.
#[allow(clippy::too_many_arguments)]
pub(crate) fn solve_ridge<F: SKBackendSelection>(
    configured: SKRidgeSolver,
    representation: &'static str,
    include_intercept: bool,
    gram: &SKCenteredGramState<F>,
    dense: ArrayView2<F>,
    target: ArrayView1<F>,
    alphas: &[F],
    backend: SKBackendKind,
) -> Result<SKRidgeSolution<F>, SKError> {
    let (centered_gram, centered_cross, feature_means, target_mean) = gram.centered();
    let solve_design = if include_intercept {
        center_design(dense, feature_means.view())
    } else {
        dense.to_owned()
    };
    let solve_target = if include_intercept {
        center_target(target, target_mean)
    } else {
        target.to_owned()
    };
    let (solve_gram, solve_cross) = if include_intercept {
        (centered_gram.view(), centered_cross.view())
    } else {
        (gram.gram.view(), gram.cross.view())
    };
    let inputs = SKRidgeSolveInputs {
        gram: solve_gram,
        cross: solve_cross,
        design: solve_design.view(),
        target: solve_target.view(),
        alphas,
    };
    let (resolved, result) = resolve_and_solve(configured, representation, inputs, backend)?;
    let intercepts = if include_intercept {
        result
            .coefficients
            .outer_iter()
            .map(|row| target_mean - row.dot(&feature_means))
            .collect()
    } else {
        Array1::zeros(result.coefficients.nrows())
    };
    Ok(SKRidgeSolution {
        resolved,
        coefficients: result.coefficients,
        intercepts,
        rank: result.rank,
        singular_values: result.singular_values,
    })
}

/// Resolve the solver for the representation and run it, returning the resolved
/// solver and its solution.
fn resolve_and_solve<F: SKBackendSelection>(
    configured: SKRidgeSolver,
    representation: &'static str,
    inputs: SKRidgeSolveInputs<'_, F>,
    backend: SKBackendKind,
) -> Result<(SKRidgeSolver, SKRidgeSolveResult<F>), SKError> {
    if representation != "dense" {
        return Err(reject_sparse(configured));
    }
    match resolve_solver(configured, representation) {
        SKRidgeSolver::Cholesky => solve_cholesky_with_fallback(inputs, backend),
        SKRidgeSolver::Svd => Ok((SKRidgeSolver::Svd, solve_svd(backend, &inputs)?)),
        SKRidgeSolver::SparseCg | SKRidgeSolver::Automatic => {
            Err(SKError::UnsupportedRepresentation {
                representation: "dense",
                suggestion: "select Cholesky or Svd for dense input; SparseCg requires sparse input",
            })
        }
    }
}

/// Cholesky per `alpha`, falling back to `Svd` when every `alpha` is zero or the
/// factor fails.
fn solve_cholesky_with_fallback<F: SKBackendSelection>(
    inputs: SKRidgeSolveInputs<'_, F>,
    backend: SKBackendKind,
) -> Result<(SKRidgeSolver, SKRidgeSolveResult<F>), SKError> {
    let every_alpha_is_zero = inputs.alphas.iter().all(|alpha| *alpha == F::zero());
    if !every_alpha_is_zero {
        if let Ok(result) = solve_cholesky(backend, &inputs) {
            return Ok((SKRidgeSolver::Cholesky, result));
        }
    }
    Ok((SKRidgeSolver::Svd, solve_svd(backend, &inputs)?))
}

/// Solve `(Ĉ + αI)β = ĉ` once per `alpha` through the backend's Cholesky.
fn solve_cholesky<F: SKBackendSelection>(
    backend: SKBackendKind,
    inputs: &SKRidgeSolveInputs<'_, F>,
) -> Result<SKRidgeSolveResult<F>, SKError> {
    let features_count = inputs.gram.ncols();
    let mut coefficients = Array2::zeros((inputs.alphas.len(), features_count));
    for (row_index, &alpha) in inputs.alphas.iter().enumerate() {
        let mut system = inputs.gram.to_owned();
        system.diag_mut().mapv_inplace(|value| value + alpha);
        let beta = solve_with_cholesky(backend, system.view(), inputs.cross)?;
        coefficients.row_mut(row_index).assign(&beta);
    }
    Ok(SKRidgeSolveResult {
        coefficients,
        rank: features_count,
        singular_values: Vec::new(),
    })
}

/// Solve every `alpha` from one SVD of the (centered) design.
fn solve_svd<F: SKBackendSelection>(
    backend: SKBackendKind,
    inputs: &SKRidgeSolveInputs<'_, F>,
) -> Result<SKRidgeSolveResult<F>, SKError> {
    let (coefficients, rank, singular_values) =
        solve_with_svd(backend, inputs.design, inputs.target, inputs.alphas)?;
    Ok(SKRidgeSolveResult {
        coefficients,
        rank,
        singular_values,
    })
}
