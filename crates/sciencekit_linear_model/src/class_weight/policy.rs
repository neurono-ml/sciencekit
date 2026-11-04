//! The class-weighting policy value (spec `stochastic-gradient-solver`,
//! decision 11).

/// The class-weighting policy of a linear classifier fit.
///
/// Shared by the logistic and the stochastic-gradient families so the balanced
/// formula and its validation live in one place.
#[derive(Debug, Clone, PartialEq)]
pub enum SKClassWeight<F> {
    /// Every sample weighs one.
    None,
    /// Weight each class inversely to its frequency.
    Balanced,
    /// An explicit weight per class, in canonical class order.
    Explicit(Vec<F>),
}

#[allow(clippy::derivable_impls)]
impl<F> Default for SKClassWeight<F> {
    fn default() -> Self {
        SKClassWeight::None
    }
}
