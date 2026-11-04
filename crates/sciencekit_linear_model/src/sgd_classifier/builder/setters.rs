//! Every configuration knob of the SGD classifier builder (spec
//! `stochastic-gradient-classifier`).

use sciencekit_common::{SKBackendKind, SKExecutionMode, SKFloat};

use super::construction::SKSGDClassifierBuilder;
use crate::class_weight::SKClassWeight;
use crate::sgd::{SKConvergencePolicy, SKSGDClassifierLoss, SKSGDLearningRate, SKSGDPenalty};
use crate::sgd_classifier::coefficients::SKSGDClassifierCoefficients;
use crate::sgd_classifier::model::SKSGDClassifierModel;

impl<F: SKFloat> SKSGDClassifierBuilder<F> {
    /// The loss of every one-vs-rest head (default `Hinge`).
    pub fn loss(mut self, value: SKSGDClassifierLoss) -> Self {
        self.loss = value;
        self
    }

    /// The regularization penalty (default `L2`).
    pub fn penalty(mut self, value: SKSGDPenalty) -> Self {
        self.config.penalty = value;
        self
    }

    /// The regularization strength (default `1e-4`).
    pub fn alpha(mut self, value: F) -> Self {
        self.config.alpha = value;
        self
    }

    /// The elastic-net mixing parameter (default `0.15`, honored only with
    /// `ElasticNet`).
    pub fn l1_ratio(mut self, value: F) -> Self {
        self.config.l1_ratio = value;
        self
    }

    /// Whether the fit solves for an unpenalized intercept (default `true`).
    pub fn include_intercept(mut self, value: bool) -> Self {
        self.config.include_intercept = value;
        self
    }

    /// The learning-rate schedule (default `Optimal`).
    pub fn learning_rate(mut self, value: SKSGDLearningRate) -> Self {
        self.config.learning_rate = value;
        self
    }

    /// The initial learning rate (default `0.01`).
    pub fn initial_learning_rate(mut self, value: F) -> Self {
        self.config.initial_learning_rate = value;
        self
    }

    /// The inverse-scaling decay power (default `0.5`).
    pub fn inverse_scaling_power(mut self, value: F) -> Self {
        self.config.inverse_scaling_power = value;
        self
    }

    /// The convergence tolerance (default `1e-3`).
    pub fn tolerance(mut self, value: F) -> Self {
        self.config.tolerance = value;
        self
    }

    /// The in-memory epoch budget (default `1000`).
    pub fn maximum_number_of_iterations(mut self, value: usize) -> Self {
        self.config.maximum_number_of_iterations = value;
        self
    }

    /// The consecutive non-improving epochs tolerated (default `5`).
    pub fn number_of_epochs_without_improvement(mut self, value: usize) -> Self {
        self.config.number_of_epochs_without_improvement = value;
        self
    }

    /// Whether a held-out validation slice drives early stopping (default
    /// `false`).
    pub fn early_stopping(mut self, value: bool) -> Self {
        self.config.early_stopping = value;
        self
    }

    /// The fraction of rows reserved for validation (default `0.1`).
    pub fn validation_fraction(mut self, value: F) -> Self {
        self.config.validation_fraction = value;
        self
    }

    /// Whether each epoch shuffles its row order (default `true`).
    pub fn shuffle(mut self, value: bool) -> Self {
        self.config.shuffle = value;
        self
    }

    /// The seed of the shuffler (default `0`).
    pub fn random_seed(mut self, value: u64) -> Self {
        self.config.random_seed = value;
        self
    }

    /// Whether the model reports the averaged iterate (default `false`).
    pub fn average(mut self, value: bool) -> Self {
        self.config.average = value;
        self
    }

    /// The class-weighting policy (default `None`).
    pub fn class_weight(mut self, value: SKClassWeight<F>) -> Self {
        self.class_weight = value;
        self
    }

    /// Seed every head from an explicit coefficient source.
    pub fn warm_start_from(mut self, value: SKSGDClassifierCoefficients<F>) -> Self {
        self.warm_start = Some(value);
        self
    }

    /// Seed every head from an already fitted model's coefficients.
    pub fn warm_start_from_model(mut self, model: &SKSGDClassifierModel<F>) -> Self {
        self.warm_start = Some(
            SKSGDClassifierCoefficients::new(
                model.classes().clone(),
                model.coefficients(),
                model.intercepts(),
            )
            .expect("a fitted model yields valid coefficients"),
        );
        self
    }

    /// What an unconverged fit does when its budget runs out (default
    /// `Strict`).
    pub fn convergence_policy(mut self, value: SKConvergencePolicy) -> Self {
        self.config.convergence_policy = value;
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
}
