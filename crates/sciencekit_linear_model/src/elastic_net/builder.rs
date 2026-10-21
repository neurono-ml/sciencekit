//! The elastic-net configuration surface: the builder (spec `elastic-net`).

use sciencekit_common::{
    SKBackendKind, SKError, SKExecutionMode, SKFloat,
    builders::{SKBuilder, SKBuilderState},
};

use crate::coordinate_descent::SKCoordinateDescentConfig;
use crate::coordinate_descent::SKIntoAlphas;
use crate::coordinate_descent::SKSelection;
use crate::coordinate_descent::config::{
    sk_validate_coordinate_descent_hyperparameters, sk_validate_l1_ratio,
};
use crate::elastic_net::core_implementation::SKElasticNet;

/// Builder for [`SKElasticNet`]: the only construction surface.
///
/// Defaults: `alpha = [1.0]`, `l1_ratio = 0.5`, intercept included,
/// unconstrained sign, cyclic selection, screening enabled, `tolerance = 1e-4`,
/// `maximum_number_of_iterations = 1000`, backend `Faer`, execution intent
/// `Automatic`.
pub struct SKElasticNetBuilder<F> {
    pub(super) state: SKBuilderState,
    pub(super) alphas: Vec<F>,
    pub(super) l1_ratio: F,
    pub(super) include_intercept: bool,
    pub(super) positive: bool,
    pub(super) selection: SKSelection,
    pub(super) screening: bool,
    pub(super) tolerance: F,
    pub(super) maximum_number_of_iterations: usize,
    pub(super) backend: SKBackendKind,
}

impl<F: SKFloat> SKElasticNetBuilder<F> {
    /// A fresh builder with the documented defaults.
    pub fn new() -> SKElasticNetBuilder<F> {
        Self {
            state: SKBuilderState::new(),
            alphas: vec![F::one()],
            l1_ratio: F::from(0.5).expect("0.5 fits in the scalar"),
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

    /// The L1/L2 mixing ratio in the closed interval `[0, 1]` (default `0.5`).
    pub fn l1_ratio(mut self, value: F) -> Self {
        self.l1_ratio = value;
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

    /// The coordinate-descent configuration the built estimator resolves to.
    pub(crate) fn coordinate_descent_config(&self) -> SKCoordinateDescentConfig<F> {
        SKCoordinateDescentConfig {
            tolerance: self.tolerance,
            maximum_number_of_iterations: self.maximum_number_of_iterations,
            positive: self.positive,
            selection: self.selection,
            screening: self.screening,
        }
    }
}

impl<F: SKFloat> Default for SKElasticNetBuilder<F> {
    fn default() -> Self {
        Self::new()
    }
}

impl<F: SKFloat> SKBuilder<SKElasticNet<F>> for SKElasticNetBuilder<F> {
    fn execution_mode<E>(&mut self, mode: E) -> &mut Self
    where
        E: TryInto<SKExecutionMode>,
    {
        self.state.execution_mode(mode);
        self
    }

    fn build(self) -> Result<SKElasticNet<F>, SKError> {
        sk_validate_coordinate_descent_hyperparameters(
            &self.alphas,
            self.tolerance,
            self.maximum_number_of_iterations,
        )?;
        sk_validate_l1_ratio(self.l1_ratio)?;
        let config = self.coordinate_descent_config();
        Ok(SKElasticNet {
            plan: self.state,
            alphas: self.alphas,
            l1_ratio: self.l1_ratio,
            include_intercept: self.include_intercept,
            config,
            backend: self.backend,
        })
    }
}
