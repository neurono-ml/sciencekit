//! The per-alpha parallel solve: one independent coordinate-descent solve per
//! configured penalty, over the compute pool (spec `coordinate-descent-solver`).

use ndarray::{Array1, Array2, ArrayView1};
use rayon::prelude::*;
use sciencekit_common::{SKError, SKFloat};

use super::super::access::SKColumnAccess;
use super::super::config::SKCoordinateDescentConfig;
use super::super::solution::{SKCoordinateDescentAlphaSolution, SKCoordinateDescentSolution};
use super::orchestration::sk_solve_alpha;

/// Solve the coordinate-descent objective once per `alpha`, in parallel when
/// more than one penalty is configured.
///
/// Each `alpha` is an independent L1/L2-regularized problem, so the per-alpha
/// solves carry no shared state and their results are independent of the thread
/// count.
pub(crate) fn sk_solve_all_alphas<F: SKFloat, A: SKColumnAccess<F> + Sync>(
    access: &A,
    target: ArrayView1<F>,
    alphas: &[F],
    l1_ratio: F,
    include_intercept: bool,
    config: &SKCoordinateDescentConfig<F>,
) -> Result<SKCoordinateDescentSolution<F>, SKError> {
    // Each `alpha` is a component with its own derived generator, indexed by
    // its position in the configured sequence, so the parallel solve never
    // shares a stream and cannot depend on completion order.
    let solve_one = |index: usize, alpha: &F| {
        sk_solve_alpha(
            access,
            target,
            *alpha,
            l1_ratio,
            include_intercept,
            index as u64,
            config,
        )
    };
    let solutions: Vec<Result<SKCoordinateDescentAlphaSolution<F>, SKError>> = if alphas.len() > 1 {
        alphas
            .par_iter()
            .enumerate()
            .map(|(index, alpha)| solve_one(index, alpha))
            .collect()
    } else {
        alphas
            .iter()
            .enumerate()
            .map(|(index, alpha)| solve_one(index, alpha))
            .collect()
    };
    let columns = access.columns_count();
    let mut coefficients = Array2::zeros((alphas.len(), columns));
    let mut intercepts = Array1::zeros(alphas.len());
    let mut screened_count = 0usize;
    for (row_index, solution) in solutions.into_iter().enumerate() {
        let solution = solution?;
        coefficients
            .row_mut(row_index)
            .assign(&solution.coefficients);
        intercepts[row_index] = solution.intercept;
        screened_count += solution.screened_count;
    }
    Ok(SKCoordinateDescentSolution {
        coefficients,
        intercepts,
        screened_count,
    })
}
