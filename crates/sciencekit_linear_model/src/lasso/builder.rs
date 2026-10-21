//! The lasso configuration surface: the builder (spec `lasso`).

use sciencekit_common::{
    SKBackendKind, SKError, SKExecutionMode, SKFloat,
    builders::{SKBuilder, SKBuilderState},
};

use crate::coordinate_descent::SKCoordinateDescentConfig;
use crate::coordinate_descent::SKIntoAlphas;
use crate::coordinate_descent::config::sk_validate_coordinate_descent_hyperparameters;
use crate::coordinate_descent::SKSelection;
use crate::lasso::core_implementation::SKLasso;

/// Builder for [`SKLasso`]: the only construction surface.
///
/// Defaults: `alpha = [1.0]`, intercept included, unconstrained sign, cyclic
/// selection, screening enabled, `tolerance = 1e-4`,
/// `maximum_number_of_iterations = 1000`, backend `Faer`, execution intent
/// `Automatic`.
pub struct SKLassoBuilder<F> {
    pub(super) state: SKBuilderState,
    pub(super) alphas: Vec<F>,
    pub(super) include_intercept: bool,
    pub(super) positive: bool,
    pub(super) selection: SKSelection,
    pub(super) screening: bool,
    pub(super) tolerance: F,
    pub(super) maximum_number_of_iterations: usize,
    pub(super) backend: SKBackendKind,
}

impl<F: SKFloat> SKLassoBuilder<F> {
    /// A fresh builder with the documented defaults.
    pub fn new() -> SKLassoBuilder<F> {
        Self {
            state: SKBuilderState::new(),
            alphas: vec![F::one()],
            include_intercept: true,
            positive: false,
            selection: SKSelection::Cyclic,
            screening: true,
            tolerance: F::from(1e-4).expect("1e-4 fits in the scalar"),
            maximum_number_of_iterations: 1000,
            backend: SKBackendKind::Faer,
        }
    }

    /// The penalty value or sequence (default `1.0`); accepts a scalar or a
    /// sequence through [`SKIntoAlphas`].
    pub fn alpha<A: SKIntoAlphas<F>>(mut self, value: A) -> Self {
        self.alphas = value.into_alphas();
        self
    }

    /// Whether the fit solves for an unpenalized intercept (default `true`).
    pub fn include_intercept(mut self, value: bool) -> Self {
        self.include_intercept = value;
        self
    }

    /// Whether every coefficient is constrained to be non-negative (default
    /// `false`).
    pub fn positive(mut self, value: bool) -> Self {
        self.positive = value;
        self
    }

    /// The coordinate sweep order (default `Cyclic`).
    pub fn selection(mut self, value: SKSelection) -> Self {
        self.selection = value;
        self
    }

    /// Whether gap-safe screening is enabled (default `true`).
    pub fn screening(mut self, value: bool) -> Self {
        self.screening = value;
        self
    }

    /// The convergence tolerance (default `1e-4`).
    pub fn tolerance(mut self, value: F) -> Self {
        self.tolerance = value;
        self
    }

    /// The iteration budget (default `1000`).
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

impl<F: SKFloat> Default for SKLassoBuilder<F> {
    fn default() -> Self {
        Self::new()
    }
}

impl<F: SKFloat> SKBuilder<SKLasso<F>> for SKLassoBuilder<F> {
    fn execution_mode<E>(&mut self, mode: E) -> &mut Self
    where
        E: TryInto<SKExecutionMode>,
    {
        self.state.execution_mode(mode);
        self
    }

    fn build(self) -> Result<SKLasso<F>, SKError> {
        sk_validate_coordinate_descent_hyperparameters(
            &self.alphas,
            self.tolerance,
            self.maximum_number_of_iterations,
        )?;
        Ok(SKLasso {
            plan: self.state,
            alphas: self.alphas,
            include_intercept: self.include_intercept,
            positive: self.positive,
            selection: self.selection,
            screening: self.screening,
            tolerance: self.tolerance,
            maximum_number_of_iterations: self.maximum_number_of_iterations,
            backend: self.backend,
        })
    }
}

/// Assemble the coordinate-descent configuration from a lasso estimator.
pub(super) fn coordinate_descent_config<F: SKFloat>(
    tolerance: F,
    maximum_number_of_iterations: usize,
    positive: bool,
    selection: SKSelection,
    screening: bool,
) -> SKCoordinateDescentConfig<F> {
    SKCoordinateDescentConfig {
        tolerance,
        maximum_number_of_iterations,
        positive,
        selection,
        screening,
    }
}
