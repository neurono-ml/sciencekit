//! The criterion a tree optimizes.

/// The criterion a tree optimizes.
///
/// Classification supports [`SKTreeCriterion::Gini`] and
/// [`SKTreeCriterion::Entropy`]; regression supports
/// [`SKTreeCriterion::SquaredError`] and [`SKTreeCriterion::AbsoluteError`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SKTreeCriterion {
    /// Gini impurity (classification default).
    Gini,
    /// Shannon entropy (classification alternative).
    Entropy,
    /// Squared error around the leaf mean (regression default).
    SquaredError,
    /// Absolute error around the leaf median (regression alternative).
    AbsoluteError,
}
