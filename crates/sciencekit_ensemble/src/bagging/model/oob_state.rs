//! Out-of-bag score state stored on fitted bagging models.

use sciencekit_common::SKError;

/// Fitted out-of-bag state stored on a bagging model.
#[derive(Debug, Clone, PartialEq)]
#[allow(dead_code)]
pub(in crate::bagging) enum SKBaggingOobState<F> {
    Disabled,
    NoScoredRows,
    Scored { score: F, scored_rows: usize },
}

impl<F: Copy> SKBaggingOobState<F> {
    pub(in crate::bagging) fn score(&self) -> Result<F, SKError> {
        match self {
            SKBaggingOobState::Scored { score, .. } => Ok(*score),
            SKBaggingOobState::Disabled => Err(SKError::UnsupportedOperation {
                operation: "oob_score",
                reason: "out-of-bag scoring was not enabled on the builder".to_string(),
            }),
            SKBaggingOobState::NoScoredRows => Err(SKError::UnsupportedOperation {
                operation: "oob_score",
                reason: "no training rows fell out of bag across the fitted estimators".to_string(),
            }),
        }
    }

    pub(in crate::bagging) fn scored_rows(&self) -> usize {
        match self {
            SKBaggingOobState::Scored { scored_rows, .. } => *scored_rows,
            SKBaggingOobState::Disabled | SKBaggingOobState::NoScoredRows => 0,
        }
    }
}
