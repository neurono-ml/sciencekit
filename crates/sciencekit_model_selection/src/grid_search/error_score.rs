//! The candidate-failure policy of a grid search (spec `grid-search`).

use sciencekit_common::SKFloat;

/// How a failing candidate/fold contributes to the cross-validation result.
///
/// `Raise` (the default) propagates the candidate's structured error;
/// `Value(value)` records the value for the failing fold and keeps evaluating.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum SKErrorScore<F: SKFloat> {
    /// Propagate the first failing fold's error.
    #[default]
    Raise,
    /// Record this value for the failing fold and continue.
    Value(F),
}

impl<F: SKFloat> SKErrorScore<F> {
    /// Records a numeric value for failing folds instead of raising.
    pub fn value(value: F) -> Self {
        SKErrorScore::Value(value)
    }

    /// Whether failures are propagated.
    pub fn is_raise(&self) -> bool {
        matches!(self, SKErrorScore::Raise)
    }
}
