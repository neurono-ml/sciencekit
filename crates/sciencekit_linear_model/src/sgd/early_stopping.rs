//! The held-out validation slice and the best-iterate bookkeeping (spec
//! `stochastic-gradient-solver`, decision 7).

use ndarray::{Array1, ArrayView1};
use sciencekit_common::SKFloat;

/// Split the row indices into a deterministic validation prefix and the
/// training remainder.
///
/// The reserved prefix is never shuffled together with training. With fewer
/// than two rows there is nothing to hold out and the validation slice is
/// empty.
pub(crate) fn sk_split_validation_indices<F: SKFloat>(
    number_of_rows: usize,
    validation_fraction: F,
) -> (Vec<usize>, Vec<usize>) {
    if number_of_rows < 2 {
        return ((0..number_of_rows).collect(), Vec::new());
    }
    let fraction = validation_fraction
        .to_f64()
        .expect("the float bound is representable as f64");
    let scaled = (fraction * number_of_rows as f64).round() as usize;
    let number_of_validation = scaled.clamp(1, number_of_rows - 1);
    let validation = (0..number_of_validation).collect();
    let training = (number_of_validation..number_of_rows).collect();
    (training, validation)
}

/// Keeps the best validation iterate and counts the non-improving epochs.
#[derive(Debug, Clone)]
pub(crate) struct SKEarlyStopping<F> {
    tolerance: F,
    patience: usize,
    best_weights: Option<Array1<F>>,
    best_intercept: Option<F>,
    best_objective: Option<F>,
    epochs_without_improvement: usize,
}

impl<F: SKFloat> SKEarlyStopping<F> {
    /// Start from the configured tolerance and patience.
    pub(crate) fn new(tolerance: F, patience: usize) -> Self {
        Self {
            tolerance,
            patience,
            best_weights: None,
            best_intercept: None,
            best_objective: None,
            epochs_without_improvement: 0,
        }
    }

    /// Score one epoch, keeping the best iterate; returns whether to stop.
    pub(crate) fn record(&mut self, weights: ArrayView1<F>, intercept: F, objective: F) -> bool {
        let improved = self
            .best_objective
            .is_none_or(|best| objective < best - self.tolerance);
        if improved {
            self.best_weights = Some(weights.to_owned());
            self.best_intercept = Some(intercept);
            self.best_objective = Some(objective);
            self.epochs_without_improvement = 0;
            return false;
        }
        self.epochs_without_improvement += 1;
        self.epochs_without_improvement >= self.patience
    }

    /// The coefficients of the best validation epoch.
    pub(crate) fn best_weights(&self) -> Option<ArrayView1<F>> {
        self.best_weights.as_ref().map(|weights| weights.view())
    }

    /// The intercept of the best validation epoch.
    pub(crate) fn best_intercept(&self) -> Option<F> {
        self.best_intercept
    }

    /// The best validation objective.
    pub(crate) fn best_objective(&self) -> Option<F> {
        self.best_objective
    }

    /// The consecutive non-improving epochs observed so far.
    pub(crate) fn epochs_without_improvement(&self) -> usize {
        self.epochs_without_improvement
    }
}
