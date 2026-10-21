//! The shared learned state of `SKLassoModel` and `SKElasticNetModel`
//! (spec `coordinate-descent-solver`, Decisions 1 and 2).
//!
//! `SKLassoModel` and `SKElasticNetModel` are distinct public types wrapping
//! this internal core: the learned coefficients, intercepts, backend kind and
//! the retained design that backs incremental updates. The core is the single
//! implementation so the two estimators cannot drift.

use ndarray::{Array1, Array2};
use sciencekit_common::{SKBackendKind, SKExecutionMode, SKFloat};

use super::super::accumulator::SKCoordinateDescentState;
use super::super::config::SKCoordinateDescentConfig;
use super::super::solution::SKCoordinateDescentSolution;

/// The shared learned state of the L1-regularized fitted models.
#[derive(Debug, Clone)]
pub(crate) struct SKCoordinateDescentModelCore<F> {
    pub(crate) alphas: Vec<F>,
    pub(crate) l1_ratio: F,
    pub(crate) coefficients: Array2<F>,
    pub(crate) intercepts: Array1<F>,
    pub(crate) include_intercept: bool,
    pub(crate) backend: SKBackendKind,
    pub(crate) execution_intent: SKExecutionMode,
    pub(crate) screened_count: usize,
    pub(crate) config: SKCoordinateDescentConfig<F>,
    pub(crate) state: SKCoordinateDescentState<F>,
}

impl<F: SKFloat> SKCoordinateDescentModelCore<F> {
    /// Assemble the core from a solve and the retained design.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn from_solution(
        alphas: Vec<F>,
        l1_ratio: F,
        solution: SKCoordinateDescentSolution<F>,
        include_intercept: bool,
        backend: SKBackendKind,
        execution_intent: SKExecutionMode,
        config: SKCoordinateDescentConfig<F>,
        state: SKCoordinateDescentState<F>,
    ) -> Self {
        Self {
            alphas,
            l1_ratio,
            coefficients: solution.coefficients,
            intercepts: solution.intercepts,
            include_intercept,
            backend,
            execution_intent,
            screened_count: solution.screened_count,
            config,
            state,
        }
    }
}
