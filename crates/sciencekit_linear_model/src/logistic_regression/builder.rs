//! The logistic-regression configuration surface: the builder (spec
//! `logistic-regression`).

use sciencekit_common::{
    SKBackendKind, SKError, SKExecutionMode, SKFloat,
    builders::{SKBuilder, SKBuilderState},
};

use crate::logistic::SKLogisticSolution;
use crate::logistic::config::{
    SKClassWeight, SKLogisticConfig, SKLogisticPenalty, SKLogisticSolver, SKMultiClass,
    sk_validate_logistic_hyperparameters,
};
use crate::logistic::resolution::sk_validate_solver_penalty;
use crate::logistic_regression::core_implementation::SKLogisticRegression;

/// Builder for [`SKLogisticRegression`]: the only construction surface.
pub struct SKLogisticRegressionBuilder<F> {
    pub(super) state: SKBuilderState,
    pub(super) penalty: SKLogisticPenalty,
    pub(super) l1_ratio: F,
    pub(super) alpha: F,
    pub(super) c_was_set: bool,
    pub(super) alpha_was_set: bool,
    pub(super) solver: SKLogisticSolver,
    pub(super) multi_class: SKMultiClass,
    pub(super) class_weight: SKClassWeight<F>,
    pub(super) warm_start: bool,
    pub(super) warm_start_solution: Option<SKLogisticSolution<F>>,
    pub(super) include_intercept: bool,
    pub(super) tolerance: F,
    pub(super) maximum_number_of_iterations: usize,
    pub(super) backend: SKBackendKind,
}

impl<F: SKFloat> SKLogisticRegressionBuilder<F> {
    /// A fresh builder with the documented defaults.
    pub fn new() -> SKLogisticRegressionBuilder<F> {
        Self {
            state: SKBuilderState::new(),
            penalty: SKLogisticPenalty::L2,
            l1_ratio: F::from(0.5).expect("0.5 fits"),
            alpha: F::one(),
            c_was_set: false,
            alpha_was_set: false,
            solver: SKLogisticSolver::Lbfgs,
            multi_class: SKMultiClass::Auto,
            class_weight: SKClassWeight::None,
            warm_start: false,
            warm_start_solution: None,
            include_intercept: true,
            tolerance: F::from(1e-4).expect("1e-4 fits"),
            maximum_number_of_iterations: 100,
            backend: SKBackendKind::Faer,
        }
    }

    /// The regularization penalty (default `L2`).
    pub fn penalty(mut self, value: SKLogisticPenalty) -> Self {
        self.penalty = value;
        self
    }

    /// The elastic-net mixing parameter (default `0.5`, honored only with
    /// `ElasticNet`).
    pub fn l1_ratio(mut self, value: F) -> Self {
        self.l1_ratio = value;
        self
    }

    /// The inverse regularization strength (default `1.0`, `alpha = 1/C`).
    #[allow(non_snake_case)]
    pub fn C(mut self, value: F) -> Self {
        if self.alpha_was_set {
            tracing::warn!("both C and alpha were set; the last setter wins");
        }
        self.alpha = F::one() / value;
        self.c_was_set = true;
        self
    }

    /// The regularization strength (default `1.0`, the reciprocal of `C`).
    pub fn alpha(mut self, value: F) -> Self {
        if self.c_was_set {
            tracing::warn!("both C and alpha were set; the last setter wins");
        }
        self.alpha = value;
        self.alpha_was_set = true;
        self
    }

    /// The optimization solver (default `Lbfgs`).
    pub fn solver(mut self, value: SKLogisticSolver) -> Self {
        self.solver = value;
        self
    }

    /// The multi-class strategy (default `Auto`).
    pub fn multi_class(mut self, value: SKMultiClass) -> Self {
        self.multi_class = value;
        self
    }

    /// The class-weighting policy (default `None`).
    pub fn class_weight(mut self, value: SKClassWeight<F>) -> Self {
        self.class_weight = value;
        self
    }

    /// Whether the optimizer is seeded from a supplied solution (default
    /// `false`).
    pub fn warm_start(mut self, value: bool) -> Self {
        self.warm_start = value;
        self
    }

    /// Seed the optimizer from a previous fitted model's solution.
    pub fn warm_start_from(
        mut self,
        model: &crate::logistic_regression::model::SKLogisticRegressionModel<F>,
    ) -> Self {
        self.warm_start_solution = Some(SKLogisticSolution {
            coefficients: model.coefficients().to_owned(),
            intercepts: model.intercepts().to_owned(),
        });
        self.warm_start = true;
        self
    }

    /// Whether the fit solves for an unpenalized intercept (default `true`).
    pub fn include_intercept(mut self, value: bool) -> Self {
        self.include_intercept = value;
        self
    }

    /// The convergence tolerance (default `1e-4`).
    pub fn tolerance(mut self, value: F) -> Self {
        self.tolerance = value;
        self
    }

    /// The iteration budget (default `100`).
    pub fn maximum_number_of_iterations(mut self, value: usize) -> Self {
        self.maximum_number_of_iterations = value;
        self
    }

    /// The math backend kind (default `SKBackendKind::Faer`).
    pub fn backend(mut self, value: SKBackendKind) -> Self {
        self.backend = value;
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

    /// Assemble the objective configuration.
    pub(super) fn logistic_config(&self) -> SKLogisticConfig<F> {
        SKLogisticConfig {
            alpha: self.alpha,
            tolerance: self.tolerance,
            maximum_number_of_iterations: self.maximum_number_of_iterations,
            penalty: self.penalty,
            l1_ratio: self.l1_ratio,
            solver: self.solver,
            multi_class: self.multi_class,
            class_weight: self.class_weight.clone(),
            include_intercept: self.include_intercept,
            warm_start: self.warm_start,
        }
    }
}

impl<F: SKFloat> Default for SKLogisticRegressionBuilder<F> {
    fn default() -> Self {
        Self::new()
    }
}

impl<F: SKFloat> SKBuilder<SKLogisticRegression<F>> for SKLogisticRegressionBuilder<F> {
    fn execution_mode<E>(&mut self, mode: E) -> &mut Self
    where
        E: TryInto<SKExecutionMode>,
    {
        self.state.execution_mode(mode);
        self
    }

    fn build(self) -> Result<SKLogisticRegression<F>, SKError> {
        sk_validate_logistic_hyperparameters(
            self.alpha,
            self.tolerance,
            self.maximum_number_of_iterations,
            self.l1_ratio,
        )?;
        sk_validate_solver_penalty(self.solver, self.penalty)?;
        let config = self.logistic_config();
        Ok(SKLogisticRegression {
            plan: self.state,
            config,
            warm_start_solution: self.warm_start_solution,
            backend: self.backend,
        })
    }
}
