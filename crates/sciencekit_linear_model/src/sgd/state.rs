//! The mutable learning state the epoch drivers advance (spec
//! `stochastic-gradient-solver`, decisions 5 and 9).

use ndarray::{Array1, ArrayView1};
use sciencekit_common::SKFloat;

use super::averaging::SKAveragedIterate;
use super::learning_rate::schedule::SKLearningRateSchedule;

/// The only mutable object of the substrate: coefficients, intercept, the
/// monotone update counter, the learning-rate schedule and the optional
/// averaged iterate.
#[derive(Debug, Clone)]
pub(crate) struct SKLearningState<F> {
    /// The last-iterate coefficients.
    pub(crate) weights: Array1<F>,
    /// The last-iterate intercept.
    pub(crate) intercept: F,
    /// The total number of updates applied so far.
    pub(crate) step: usize,
    schedule: SKLearningRateSchedule<F>,
    average: Option<SKAveragedIterate<F>>,
}

impl<F: SKFloat> SKLearningState<F> {
    /// Start a learning state at the given iterate.
    pub(crate) fn new(
        weights: Array1<F>,
        intercept: F,
        schedule: SKLearningRateSchedule<F>,
        average: bool,
    ) -> Self {
        let average = average.then(|| SKAveragedIterate::new(weights.clone(), intercept));
        Self {
            weights,
            intercept,
            step: 0,
            schedule,
            average,
        }
    }

    /// The learning rate for the current update count.
    pub(crate) fn learning_rate(&self) -> F {
        self.schedule.rate(self.step)
    }

    /// Halve the adaptive rate after a non-improving epoch.
    pub(crate) fn halve_learning_rate(&mut self) {
        self.schedule.halve();
    }

    /// Move the last iterate, and the origin of the average, to a chosen
    /// iterate.
    ///
    /// Early stopping uses this so the state the model continues from is
    /// exactly the best-validation iterate the model reports.
    pub(crate) fn reset_iterate(&mut self, weights: ArrayView1<F>, intercept: F) {
        self.weights.assign(&weights);
        self.intercept = intercept;
        if let Some(average) = &mut self.average {
            average.reset(weights, intercept);
        }
    }

    /// Fold the current iterate into the running average, when enabled.
    pub(crate) fn record_visited_iterate(&mut self) {
        if let Some(average) = &mut self.average {
            average.record(self.weights.view(), self.intercept);
        }
    }

    /// The coefficients the model reports: the averaged iterate when enabled,
    /// otherwise the last iterate.
    pub(crate) fn reported_weights(&self) -> ArrayView1<F> {
        match &self.average {
            Some(average) => average.weights(),
            None => self.weights.view(),
        }
    }

    /// The intercept the model reports, matching [`Self::reported_weights`].
    pub(crate) fn reported_intercept(&self) -> F {
        match &self.average {
            Some(average) => average.intercept(),
            None => self.intercept,
        }
    }

    /// The number of iterates folded into the average (zero without averaging).
    #[cfg(test)]
    pub(crate) fn average_count(&self) -> usize {
        self.average.as_ref().map_or(0, |average| average.count())
    }
}
