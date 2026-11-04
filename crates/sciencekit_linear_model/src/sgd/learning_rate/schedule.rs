//! The deterministic learning-rate schedules (spec
//! `stochastic-gradient-solver`, decision 4).

use sciencekit_common::SKFloat;

use super::policy::SKSGDLearningRate;

/// A learning-rate schedule driven by the monotone update counter.
///
/// `Adaptive` holds a current rate that the convergence bookkeeping halves
/// after a non-improving epoch; every other schedule is a pure function of the
/// counter.
#[derive(Debug, Clone)]
pub(crate) struct SKLearningRateSchedule<F> {
    policy: SKSGDLearningRate,
    initial_learning_rate: F,
    inverse_scaling_power: F,
    alpha: F,
    optimal_offset: F,
    adaptive_learning_rate: F,
}

impl<F: SKFloat> SKLearningRateSchedule<F> {
    /// Build a schedule from a validated configuration.
    pub(crate) fn new(
        policy: SKSGDLearningRate,
        initial_learning_rate: F,
        inverse_scaling_power: F,
        alpha: F,
        optimal_offset: F,
    ) -> Self {
        Self {
            policy,
            initial_learning_rate,
            inverse_scaling_power,
            alpha,
            optimal_offset,
            adaptive_learning_rate: initial_learning_rate,
        }
    }

    /// The selected schedule.
    pub(crate) fn policy(&self) -> SKSGDLearningRate {
        self.policy
    }

    /// The learning rate for a total update count of `step`.
    pub(crate) fn rate(&self, step: usize) -> F {
        let step = F::from(step).expect("the update counter fits in the scalar");
        match self.policy {
            SKSGDLearningRate::Constant => self.initial_learning_rate,
            SKSGDLearningRate::InverseScaling => {
                self.initial_learning_rate / (F::one() + step).powf(self.inverse_scaling_power)
            }
            SKSGDLearningRate::Optimal => F::one() / (self.alpha * (step + self.optimal_offset)),
            SKSGDLearningRate::Adaptive => self.adaptive_learning_rate,
        }
    }

    /// Halve the current rate on a plateau.
    ///
    /// Only `Adaptive` reacts; the other schedules ignore the signal so the
    /// caller can invoke it unconditionally.
    pub(crate) fn halve(&mut self) {
        if self.policy == SKSGDLearningRate::Adaptive {
            self.adaptive_learning_rate = self.adaptive_learning_rate / F::from(2).expect("two");
        }
    }

    /// The current adaptive rate (the initial rate for the other schedules).
    pub(crate) fn current_learning_rate(&self) -> F {
        self.adaptive_learning_rate
    }
}
