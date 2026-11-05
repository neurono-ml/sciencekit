//! Streaming classification accumulator.

use std::collections::{BTreeMap, BTreeSet};

use sciencekit_common::{SKError, SKFloat};

use crate::classification::{SKConfusionMatrix, SKLabelSpace};

/// A mergeable accumulator of classification label pairs.
///
/// Memory is bounded by the number of observed label pairs, at most the square
/// of the class count. Accumulate batches or parallel partitions and merge them
/// before finalizing the confusion matrix.
#[derive(Debug, Clone, Default)]
pub struct SKClassificationAccumulator {
    counts: BTreeMap<(i64, i64), usize>,
    observations: usize,
}

impl SKClassificationAccumulator {
    /// An empty accumulator.
    pub fn new() -> Self {
        Self::default()
    }

    /// Observe one true/predicted label pair.
    pub fn observe(&mut self, true_label: i64, predicted_label: i64) {
        *self
            .counts
            .entry((true_label, predicted_label))
            .or_insert(0) += 1;
        self.observations += 1;
    }

    /// Merge another accumulator into this one.
    pub fn merge(&mut self, other: &Self) {
        for (pair, count) in other.counts.iter() {
            *self.counts.entry(*pair).or_insert(0) += count;
        }
        self.observations += other.observations;
    }

    /// The number of observations accumulated.
    pub fn number_of_observations(&self) -> usize {
        self.observations
    }

    /// The accuracy over the accumulated label pairs.
    pub fn accuracy<F: SKFloat>(&self) -> Result<F, SKError> {
        if self.observations == 0 {
            return Err(SKError::EmptyInput {
                operation: "classification_accumulator",
            });
        }
        let correct: usize = self
            .counts
            .iter()
            .filter(|((true_label, predicted_label), _)| true_label == predicted_label)
            .map(|(_, count)| *count)
            .sum();
        let correct = num_traits::cast::<usize, F>(correct).expect("counts fit in the scalar");
        let total =
            num_traits::cast::<usize, F>(self.observations).expect("counts fit in the scalar");
        Ok(correct / total)
    }

    /// Finalize the confusion matrix over the resolved label space.
    pub fn confusion_matrix(
        &self,
        explicit_labels: Option<&[i64]>,
    ) -> Result<SKConfusionMatrix, SKError> {
        if self.observations == 0 {
            return Err(SKError::EmptyInput {
                operation: "classification_accumulator",
            });
        }
        let labels = match explicit_labels {
            Some(labels) => {
                let space = SKLabelSpace::from_explicit(labels)?;
                for (true_label, predicted_label) in self.counts.keys() {
                    space.index_of(*true_label)?;
                    space.index_of(*predicted_label)?;
                }
                space.labels().to_vec()
            }
            None => {
                let mut observed = BTreeSet::new();
                for (true_label, predicted_label) in self.counts.keys() {
                    observed.insert(*true_label);
                    observed.insert(*predicted_label);
                }
                observed.into_iter().collect()
            }
        };
        let classes = labels.len();
        let mut counts = vec![0_usize; classes * classes];
        for ((true_label, predicted_label), count) in self.counts.iter() {
            let true_index = labels
                .binary_search(true_label)
                .expect("every observed label is resolved");
            let predicted_index = labels
                .binary_search(predicted_label)
                .expect("every observed label is resolved");
            counts[true_index * classes + predicted_index] += count;
        }
        Ok(SKConfusionMatrix::from_counts(labels, counts))
    }
}
