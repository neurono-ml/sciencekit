//! Shared export helpers for the ensemble models.

use sciencekit_tree::SKTreeCriterion;

/// The stable string for a split criterion.
pub(crate) fn criterion_name(criterion: SKTreeCriterion) -> &'static str {
    match criterion {
        SKTreeCriterion::Gini => "gini",
        SKTreeCriterion::Entropy => "entropy",
        SKTreeCriterion::SquaredError => "squared_error",
        SKTreeCriterion::AbsoluteError => "absolute_error",
    }
}

/// The split criterion for its stable string.
pub(crate) fn criterion_from_name(name: &str) -> SKTreeCriterion {
    match name {
        "entropy" => SKTreeCriterion::Entropy,
        "squared_error" => SKTreeCriterion::SquaredError,
        "absolute_error" => SKTreeCriterion::AbsoluteError,
        _ => SKTreeCriterion::Gini,
    }
}
