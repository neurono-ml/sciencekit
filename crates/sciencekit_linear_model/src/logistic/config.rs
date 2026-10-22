//! The logistic objective configuration, its public enums and the shared
//! validation (spec `logistic-solver`).

use sciencekit_common::builders::sk_validate_hyperparameter;
use sciencekit_common::{SKError, SKFloat};

/// The regularization penalty of a logistic fit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SKLogisticPenalty {
    /// The L2 (ridge) penalty `‖w‖₂²`.
    L2,
    /// The L1 (lasso) penalty `‖w‖₁`, promoting exact zeros.
    L1,
    /// The mixed L1/L2 (elastic-net) penalty, controlled by `l1_ratio`.
    ElasticNet,
    /// No penalty.
    None,
}

/// The optimization solver of a logistic fit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SKLogisticSolver {
    /// Limited-memory BFGS with an orthant-wise step (the smooth default).
    Lbfgs,
    /// The LIBLINEAR dual coordinate-descent solver (one-vs-rest).
    Liblinear,
    /// The SAGA variance-reduced incremental-gradient solver.
    Saga,
}

/// How a multi-class problem is reduced to binary fits.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SKMultiClass {
    /// Resolve automatically from the solver and the class count.
    Auto,
    /// Fit one binary model per class.
    OneVersusRest,
    /// Fit a single joint softmax model.
    Multinomial,
}

/// The class-weighting policy of a logistic fit.
#[derive(Debug, Clone, PartialEq)]
pub enum SKClassWeight<F> {
    /// Every sample weighs one.
    None,
    /// Weight each class inversely to its frequency.
    Balanced,
    /// An explicit weight per class, in canonical class order.
    Explicit(Vec<F>),
}

#[allow(clippy::derivable_impls)]
impl<F> Default for SKClassWeight<F> {
    fn default() -> Self {
        SKClassWeight::None
    }
}

/// The objective configuration shared by the logistic estimators.
///
/// The regularization strength is stored as `alpha` (the reciprocal of
/// scikit-learn's `C`); the builders resolve `C`/`alpha` before constructing it.
#[derive(Debug, Clone)]
pub(crate) struct SKLogisticConfig<F> {
    /// The regularization strength `alpha = 1/C`.
    pub(crate) alpha: F,
    /// The convergence tolerance.
    pub(crate) tolerance: F,
    /// The iteration budget before declaring non-convergence.
    pub(crate) maximum_number_of_iterations: usize,
    /// The penalty.
    pub(crate) penalty: SKLogisticPenalty,
    /// The elastic-net mixing parameter (honored only with `ElasticNet`).
    pub(crate) l1_ratio: F,
    /// The solver.
    pub(crate) solver: SKLogisticSolver,
    /// The multi-class strategy.
    pub(crate) multi_class: SKMultiClass,
    /// The class-weighting policy.
    pub(crate) class_weight: SKClassWeight<F>,
    /// Whether the fit solves for an unpenalized intercept.
    pub(crate) include_intercept: bool,
    /// Whether the optimizer is seeded from a supplied solution.
    pub(crate) warm_start: bool,
}

/// Validate the numeric logistic hyperparameters: the regularization strength,
/// the tolerance, the iteration budget and the `l1_ratio` range.
pub(crate) fn sk_validate_logistic_hyperparameters<F: SKFloat>(
    alpha: F,
    tolerance: F,
    maximum_number_of_iterations: usize,
    l1_ratio: F,
) -> Result<(), SKError> {
    sk_validate_hyperparameter(
        "C",
        alpha > F::zero() && alpha.is_finite(),
        "the regularization strength must be positive and finite",
    )?;
    sk_validate_hyperparameter("tolerance", tolerance > F::zero(), "must be positive")?;
    sk_validate_hyperparameter(
        "maximum_number_of_iterations",
        maximum_number_of_iterations > 0,
        "must be at least 1",
    )?;
    sk_validate_hyperparameter(
        "l1_ratio",
        l1_ratio >= F::zero() && l1_ratio <= F::one(),
        "must lie in the closed interval [0, 1]",
    )
}
