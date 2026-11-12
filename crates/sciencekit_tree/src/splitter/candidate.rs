//! A split candidate and its impurity decrease.

use sciencekit_common::SKFloat;

/// A candidate split found by the split search.
#[derive(Debug, Clone, PartialEq)]
pub struct SKSplitCandidate<F> {
    pub(crate) feature: usize,
    pub(crate) threshold: F,
    pub(crate) impurity_decrease: F,
}

impl<F: SKFloat> SKSplitCandidate<F> {
    /// Build a candidate.
    pub(crate) fn new(feature: usize, threshold: F, impurity_decrease: F) -> Self {
        SKSplitCandidate {
            feature,
            threshold,
            impurity_decrease,
        }
    }

    /// The feature this candidate splits on.
    pub fn feature(&self) -> usize {
        self.feature
    }

    /// The threshold: values at or below it go left.
    pub fn threshold(&self) -> F {
        self.threshold
    }

    /// The impurity decrease this split achieves.
    pub fn impurity_decrease(&self) -> F {
        self.impurity_decrease
    }
}
