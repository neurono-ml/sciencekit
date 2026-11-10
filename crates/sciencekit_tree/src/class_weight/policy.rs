//! The class-weighting policy value.

/// How the classifier weighs training samples by class.
#[derive(Debug, Clone, PartialEq)]
pub enum SKClassWeight<F> {
    /// Every sample weighs one (default).
    None,
    /// Weight each class inversely to its frequency.
    Balanced,
    /// An explicit weight per class, in canonical class order.
    Explicit(Vec<F>),
}

impl<F> Default for SKClassWeight<F> {
    fn default() -> Self {
        SKClassWeight::None
    }
}
