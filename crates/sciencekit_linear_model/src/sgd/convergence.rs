//! The convergence policy and the epoch bookkeeping (spec
//! `stochastic-gradient-solver`, decision 7).

use sciencekit_common::{SKError, SKFloat};

/// What an unconverged fit does when its budget is exhausted.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SKConvergencePolicy {
    /// Return [`SKError::NotConverged`] when the full budget runs out.
    #[default]
    Strict,
    /// Return the last iterate with `converged() == false` and a warning.
    Permissive,
}

/// The plateau detector: a fit converges when the epoch objective improves by
/// no more than `tolerance` for `patience` consecutive epochs.
#[derive(Debug, Clone)]
pub(crate) struct SKConvergenceTracker<F> {
    tolerance: F,
    patience: usize,
    previous_objective: Option<F>,
    best_objective: Option<F>,
    epochs_without_improvement: usize,
    converged: bool,
}

impl<F: SKFloat> SKConvergenceTracker<F> {
    /// Start tracking from the configured tolerance and patience.
    pub(crate) fn new(tolerance: F, patience: usize) -> Self {
        Self {
            tolerance,
            patience,
            previous_objective: None,
            best_objective: None,
            epochs_without_improvement: 0,
            converged: false,
        }
    }

    /// Record one epoch's objective, returning whether the fit just converged.
    pub(crate) fn record(&mut self, objective: F) -> bool {
        let improved = match self.previous_objective {
            None => true,
            Some(previous) => previous - objective > self.tolerance,
        };
        self.previous_objective = Some(objective);
        if self.best_objective.is_none_or(|best| objective < best) {
            self.best_objective = Some(objective);
        }
        if improved {
            self.epochs_without_improvement = 0;
        } else {
            self.epochs_without_improvement += 1;
            if self.epochs_without_improvement >= self.patience {
                self.converged = true;
            }
        }
        self.converged
    }

    /// Whether the plateau criterion has been met.
    #[cfg(test)]
    pub(crate) fn converged(&self) -> bool {
        self.converged
    }

    /// The consecutive non-improving epochs observed so far.
    #[cfg(test)]
    pub(crate) fn epochs_without_improvement(&self) -> usize {
        self.epochs_without_improvement
    }
}

/// Decide what an ended fit returns.
///
/// `provider_limited` marks a one-shot source that supplied fewer passes than
/// the budget: that is never a budget exhaustion, so it never raises even under
/// [`SKConvergencePolicy::Strict`]. An early-stopped fit also stops before the
/// budget and returns its best iterate. Only a fit that executed the whole
/// requested budget can raise.
pub(crate) fn sk_finish_epochs(
    policy: SKConvergencePolicy,
    requested_budget: usize,
    epochs_taken: usize,
    converged: bool,
    provider_limited: bool,
) -> Result<bool, SKError> {
    if converged {
        return Ok(true);
    }
    if provider_limited {
        tracing::warn!(
            epochs_taken,
            "the streaming source supplied fewer passes than the requested budget; returning the last iterate without convergence"
        );
        return Ok(false);
    }
    if epochs_taken < requested_budget {
        tracing::warn!(
            epochs_taken,
            requested_budget,
            "the fit stopped before exhausting its budget (early stopping); returning the best iterate"
        );
        return Ok(false);
    }
    match policy {
        SKConvergencePolicy::Strict => Err(SKError::NotConverged {
            iterations: epochs_taken,
        }),
        SKConvergencePolicy::Permissive => {
            tracing::warn!(
                epochs_taken,
                "the fit exhausted its budget without converging; returning the last iterate"
            );
            Ok(false)
        }
    }
}
