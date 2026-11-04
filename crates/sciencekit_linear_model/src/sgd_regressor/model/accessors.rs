//! Hyperparameter accessors of the fitted SGD regressor model.

use crate::sgd::{SKConvergencePolicy, SKSGDLearningRate, SKSGDPenalty, SKSGDRegressorLoss};

use super::core_implementation::SKSGDRegressorModel;

impl<F: sciencekit_common::SKFloat> SKSGDRegressorModel<F> {
    /// The loss the model was fitted with.
    pub fn loss(&self) -> SKSGDRegressorLoss {
        self.loss
    }

    /// The `epsilon` band the loss was bound to.
    pub fn epsilon(&self) -> F {
        self.epsilon
    }

    /// The regularization penalty.
    pub fn penalty(&self) -> SKSGDPenalty {
        self.config.penalty
    }

    /// The regularization strength.
    pub fn alpha(&self) -> F {
        self.config.alpha
    }

    /// The elastic-net mixing parameter.
    pub fn l1_ratio(&self) -> F {
        self.config.l1_ratio
    }

    /// Whether the model reports an intercept.
    pub fn include_intercept(&self) -> bool {
        self.config.include_intercept
    }

    /// The learning-rate schedule.
    pub fn learning_rate(&self) -> SKSGDLearningRate {
        self.config.learning_rate
    }

    /// The configured initial learning rate.
    pub fn initial_learning_rate(&self) -> F {
        self.config.initial_learning_rate
    }

    /// The inverse-scaling decay power.
    pub fn inverse_scaling_power(&self) -> F {
        self.config.inverse_scaling_power
    }

    /// The convergence tolerance.
    pub fn tolerance(&self) -> F {
        self.config.tolerance
    }

    /// The epoch budget of an in-memory fit.
    pub fn maximum_number_of_iterations(&self) -> usize {
        self.config.maximum_number_of_iterations
    }

    /// The consecutive non-improving epochs tolerated before a plateau.
    pub fn number_of_epochs_without_improvement(&self) -> usize {
        self.config.number_of_epochs_without_improvement
    }

    /// Whether early stopping was enabled.
    pub fn early_stopping(&self) -> bool {
        self.config.early_stopping
    }

    /// The fraction of rows reserved for validation.
    pub fn validation_fraction(&self) -> F {
        self.config.validation_fraction
    }

    /// Whether each epoch shuffles its row order.
    pub fn shuffle(&self) -> bool {
        self.config.shuffle
    }

    /// The seed of the shuffler.
    pub fn random_seed(&self) -> u64 {
        self.config.random_seed
    }

    /// Whether the model reports the averaged iterate.
    pub fn average(&self) -> bool {
        self.config.average
    }

    /// What an unconverged fit did when its budget ran out.
    pub fn convergence_policy(&self) -> SKConvergencePolicy {
        self.config.convergence_policy
    }
}
