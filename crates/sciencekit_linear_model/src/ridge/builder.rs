//! The ridge configuration surface: the solver portfolio, the alpha-conversion
//! trait and the builder (spec `ridge-regression`).

use ndarray::Array1;
use sciencekit_common::{
    SKBackendKind, SKError, SKExecutionMode, SKFloat,
    builders::{SKBuilder, SKBuilderState, sk_validate_hyperparameter},
};

use crate::ridge::core_implementation::SKRidge;

/// The ridge solver portfolio.
///
/// `Automatic` resolves by input representation (dense → `Cholesky`, sparse →
/// `SparseCg`) and `Cholesky` falls back to `Svd` when every `alpha` is zero or
/// the factor fails, so the selected solver may differ from the configured one —
/// the model always exposes the solver actually used. An explicit `SparseCg`
/// applies to sparse input only; a dense input under `SparseCg` is rejected with
/// a structured representation error.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SKRidgeSolver {
    /// Resolve automatically from the input representation.
    Automatic,
    /// Centered normal equations through the backend's Cholesky factor.
    Cholesky,
    /// One centered singular-value decomposition, applied per `alpha`.
    Svd,
    /// Conjugate gradient over a borrowed sparse design, never forming the Gram.
    SparseCg,
}

/// Converts a configured penalty into the owned `alpha` sequence a ridge fit
/// consumes.
///
/// Implemented for the scalar `F`, `Vec<F>`, `Array1<F>`, `&[F]` and `[F; N]`
/// so `alpha(1.0)`, `alpha(vec![..])`, `alpha(array![..])`, `alpha(&[..])` and
/// `alpha([..])` are all accepted by the builder.
pub trait SKIntoRidgeAlphas<F> {
    /// The penalty values, in configuration order.
    fn into_alphas(self) -> Vec<F>;
}

impl<F: SKFloat> SKIntoRidgeAlphas<F> for F {
    fn into_alphas(self) -> Vec<F> {
        vec![self]
    }
}

impl<F: SKFloat> SKIntoRidgeAlphas<F> for Vec<F> {
    fn into_alphas(self) -> Vec<F> {
        self
    }
}

impl<F: SKFloat> SKIntoRidgeAlphas<F> for Array1<F> {
    fn into_alphas(self) -> Vec<F> {
        self.to_vec()
    }
}

impl<F: SKFloat> SKIntoRidgeAlphas<F> for &[F] {
    fn into_alphas(self) -> Vec<F> {
        self.to_vec()
    }
}

impl<F: SKFloat, const N: usize> SKIntoRidgeAlphas<F> for [F; N] {
    fn into_alphas(self) -> Vec<F> {
        self.into_iter().collect()
    }
}

/// Builder for [`SKRidge`]: the only construction surface.
///
/// Defaults: `alpha = [1.0]`, intercept included, `Automatic` solver,
/// `tolerance = 1e-6`, `maximum_number_of_iterations = 1000`, backend `Faer`,
/// execution intent `Automatic`.
pub struct SKRidgeBuilder<F> {
    pub(super) state: SKBuilderState,
    pub(super) alphas: Vec<F>,
    pub(super) include_intercept: bool,
    pub(super) solver: SKRidgeSolver,
    pub(super) tolerance: F,
    pub(super) maximum_number_of_iterations: usize,
    pub(super) backend: SKBackendKind,
}

impl<F: SKFloat> SKRidgeBuilder<F> {
    /// A fresh builder with the documented defaults.
    pub fn new() -> SKRidgeBuilder<F> {
        Self {
            state: SKBuilderState::new(),
            alphas: vec![F::one()],
            include_intercept: true,
            solver: SKRidgeSolver::Automatic,
            tolerance: F::from(1e-6).expect("1e-6 fits in the scalar"),
            maximum_number_of_iterations: 1000,
            backend: SKBackendKind::Faer,
        }
    }

    /// The penalty value or sequence (default `1.0`); accepts a scalar or a
    /// sequence through [`SKIntoRidgeAlphas`].
    pub fn alpha<A: SKIntoRidgeAlphas<F>>(mut self, value: A) -> Self {
        self.alphas = value.into_alphas();
        self
    }

    /// Whether the fit solves for an unpenalized intercept (default `true`).
    pub fn include_intercept(mut self, value: bool) -> Self {
        self.include_intercept = value;
        self
    }

    /// The solver portfolio selection (default `Automatic`).
    pub fn solver(mut self, solver: SKRidgeSolver) -> Self {
        self.solver = solver;
        self
    }

    /// The conjugate-gradient residual tolerance (default `1e-6`).
    pub fn tolerance(mut self, value: F) -> Self {
        self.tolerance = value;
        self
    }

    /// The conjugate-gradient iteration budget (default `1000`).
    pub fn maximum_number_of_iterations(mut self, value: usize) -> Self {
        self.maximum_number_of_iterations = value;
        self
    }

    /// The math backend kind (default `SKBackendKind::Faer`).
    pub fn backend(mut self, backend: SKBackendKind) -> Self {
        self.backend = backend;
        self
    }

    /// The execution intent (default `SKExecutionMode::Automatic`).
    pub fn execution_mode<E>(mut self, mode: E) -> Self
    where
        E: TryInto<SKExecutionMode>,
    {
        self.state.execution_mode(mode);
        self
    }
}

impl<F: SKFloat> Default for SKRidgeBuilder<F> {
    fn default() -> Self {
        Self::new()
    }
}

impl<F: SKFloat> SKBuilder<SKRidge<F>> for SKRidgeBuilder<F> {
    fn execution_mode<E>(&mut self, mode: E) -> &mut Self
    where
        E: TryInto<SKExecutionMode>,
    {
        self.state.execution_mode(mode);
        self
    }

    fn build(self) -> Result<SKRidge<F>, SKError> {
        sk_validate_hyperparameter(
            "alpha",
            !self.alphas.is_empty(),
            "must contain at least one penalty value",
        )?;
        sk_validate_hyperparameter(
            "alpha",
            self.alphas.iter().all(|alpha| *alpha >= F::zero()),
            "every penalty value must be non-negative",
        )?;
        sk_validate_hyperparameter("tolerance", self.tolerance > F::zero(), "must be positive")?;
        sk_validate_hyperparameter(
            "maximum_number_of_iterations",
            self.maximum_number_of_iterations > 0,
            "must be at least 1",
        )?;
        Ok(SKRidge {
            plan: self.state,
            alphas: self.alphas,
            include_intercept: self.include_intercept,
            solver: self.solver,
            tolerance: self.tolerance,
            maximum_number_of_iterations: self.maximum_number_of_iterations,
            backend: self.backend,
        })
    }
}
