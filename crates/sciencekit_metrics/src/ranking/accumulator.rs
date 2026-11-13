//! A mergeable accumulator for the binary ROC-AUC.
//!
//! Memory is `O(n)` in the number of observed score/label pairs: the exact
//! rank-sum AUC cannot be represented by a bounded summary, so every pair is
//! retained. A future sketch-based variant can replace this without changing
//! the public API.

use sciencekit_common::{SKError, SKFloat};

use super::roc::sk_roc_auc_score;

/// A mergeable accumulator of score/label pairs for the binary ROC-AUC.
#[derive(Debug, Clone)]
pub struct SKAucAccumulator<F> {
    pairs: Vec<(F, i64)>,
}

impl<F: SKFloat> Default for SKAucAccumulator<F> {
    fn default() -> Self {
        Self { pairs: Vec::new() }
    }
}

impl<F: SKFloat> SKAucAccumulator<F> {
    /// An empty accumulator.
    pub fn new() -> Self {
        Self::default()
    }

    /// Observe one score/label pair.
    pub fn observe(&mut self, score: F, label: i64) {
        self.pairs.push((score, label));
    }

    /// Merge another accumulator into this one by concatenation.
    pub fn merge(&mut self, other: &Self) {
        self.pairs.extend_from_slice(&other.pairs);
    }

    /// The number of observations accumulated.
    pub fn number_of_observations(&self) -> usize {
        self.pairs.len()
    }

    /// Whether no observation has been accumulated.
    pub fn is_empty(&self) -> bool {
        self.pairs.is_empty()
    }

    /// Finalize the binary AUC over every accumulated pair.
    pub fn finalize(&self, positive_label: i64) -> Result<F, SKError> {
        if self.pairs.is_empty() {
            return Err(SKError::EmptyInput {
                operation: "auc_accumulator",
            });
        }
        let mut scores = Vec::with_capacity(self.pairs.len());
        let mut labels = Vec::with_capacity(self.pairs.len());
        for (score, label) in self.pairs.iter() {
            scores.push(*score);
            labels.push(*label);
        }
        sk_roc_auc_score(&scores, &labels, positive_label)
    }
}
