//! The coordinate-descent configuration: the penalty-sequence conversion, the
//! sweep selection, the objective configuration and the shared validation
//! (spec `coordinate-descent-solver`).

use ndarray::Array1;
use sciencekit_common::builders::sk_validate_hyperparameter;
use sciencekit_common::{SKError, SKFloat};

/// The coordinate sweep order.
///
/// `Cyclic` visits the coordinates in index order on every sweep; `Random`
/// visits a deterministic pseudo-random permutation per sweep. Both converge to
/// the same solution; the selection only changes the order of the work.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SKSelection {
    /// Visit the coordinates in index order.
    Cyclic,
    /// Visit a deterministic pseudo-random permutation each sweep.
    Random,
}

/// Converts a configured penalty into the owned `alpha` sequence a
/// coordinate-descent fit consumes.
///
/// Implemented for the scalar `F`, `Vec<F>`, `Array1<F>`, `&[F]` and `[F; N]`
/// so `alpha(1.0)`, `alpha(vec![..])`, `alpha(array![..])`, `alpha(&[..])` and
/// `alpha([..])` are all accepted by the builders.
pub trait SKIntoAlphas<F> {
    /// The penalty values, in configuration order.
    fn into_alphas(self) -> Vec<F>;
}

impl<F: SKFloat> SKIntoAlphas<F> for F {
    fn into_alphas(self) -> Vec<F> {
        vec![self]
    }
}

impl<F: SKFloat> SKIntoAlphas<F> for Vec<F> {
    fn into_alphas(self) -> Vec<F> {
        self
    }
}

impl<F: SKFloat> SKIntoAlphas<F> for Array1<F> {
    fn into_alphas(self) -> Vec<F> {
        self.to_vec()
    }
}

impl<F: SKFloat> SKIntoAlphas<F> for &[F] {
    fn into_alphas(self) -> Vec<F> {
        self.to_vec()
    }
}

impl<F: SKFloat, const N: usize> SKIntoAlphas<F> for [F; N] {
    fn into_alphas(self) -> Vec<F> {
        self.into_iter().collect()
    }
}

/// The objective configuration shared by `SKLasso` and `SKElasticNet`.
///
/// The `l1_ratio` is carried by the estimator (lasso fixes it at one), so it is
/// passed to the solver separately from this configuration.
#[derive(Debug, Clone)]
pub(crate) struct SKCoordinateDescentConfig<F> {
    /// The convergence tolerance on the maximum coordinate change.
    pub(crate) tolerance: F,
    /// The sweep budget before declaring non-convergence.
    pub(crate) maximum_number_of_iterations: usize,
    /// Whether every coordinate is clamped at zero.
    pub(crate) positive: bool,
    /// The sweep order.
    pub(crate) selection: SKSelection,
    /// Whether gap-safe screening is enabled.
    pub(crate) screening: bool,
}

impl<F: SKFloat> Default for SKCoordinateDescentConfig<F> {
    fn default() -> Self {
        Self {
            tolerance: F::from(1e-4).expect("1e-4 fits in the scalar"),
            maximum_number_of_iterations: 1000,
            positive: false,
            selection: SKSelection::Cyclic,
            screening: true,
        }
    }
}

/// Validate the coordinate-descent penalty sequence, tolerance and iteration
/// budget, each with the central error taxonomy.
pub(crate) fn sk_validate_coordinate_descent_hyperparameters<F: SKFloat>(
    alphas: &[F],
    tolerance: F,
    maximum_number_of_iterations: usize,
) -> Result<(), SKError> {
    sk_validate_hyperparameter(
        "alpha",
        !alphas.is_empty(),
        "must contain at least one penalty value",
    )?;
    sk_validate_hyperparameter(
        "alpha",
        alphas.iter().all(|alpha| *alpha >= F::zero()),
        "every penalty value must be non-negative",
    )?;
    sk_validate_hyperparameter("tolerance", tolerance > F::zero(), "must be positive")?;
    sk_validate_hyperparameter(
        "maximum_number_of_iterations",
        maximum_number_of_iterations > 0,
        "must be at least 1",
    )?;
    Ok(())
}

/// Validate the elastic-net `l1_ratio` mixing parameter to the closed interval
/// `[0, 1]`.
pub(crate) fn sk_validate_l1_ratio<F: SKFloat>(l1_ratio: F) -> Result<(), SKError> {
    sk_validate_hyperparameter(
        "l1_ratio",
        l1_ratio >= F::zero() && l1_ratio <= F::one(),
        "must lie in the closed interval [0, 1]",
    )
}
