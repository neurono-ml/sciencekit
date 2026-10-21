//! Sparse ridge solver resolution and orchestration (spec `ridge-regression`).
//!
//! `Automatic` resolves to `SparseCg` for sparse input; `Cholesky` builds the
//! centered Gram from the sparse structure and solves it with the backend's
//! Cholesky factor (the shared [`solve_from_gram`] path); `Svd` is dense-only
//! and rejects sparse input.

use ndarray::ArrayView1;
use sciencekit_common::{SKBackendKind, SKError};
use sciencekit_math::SKBackendSelection;
use sprs::CsMatView;

use super::cg::solve_with_sparse_cg;
use crate::ridge::builder::SKRidgeSolver;
use crate::ridge::gram::SKCenteredGramState;
use crate::ridge::solving::{SKRidgeSolution, solve_from_gram};

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
        SKRidgeSolver::Cholesky => solve_from_gram(gram, include_intercept, alphas, backend),
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
