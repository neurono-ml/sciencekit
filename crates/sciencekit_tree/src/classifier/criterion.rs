//! The classifier's criterion implementation.

use sciencekit_common::SKFloat;

use crate::criterion::SKTreeCriterion as SKConfiguredCriterion;
use crate::growth::SKTreeCriterion;
use crate::splitter::{SKScanContext, sk_classification_impurity};

/// Weighted classification statistics backing growth and leaf outputs.
pub(crate) struct SKClassifierCriterion<'a, F> {
    pub(crate) class_indices: &'a [usize],
    pub(crate) sample_weights: &'a [F],
    pub(crate) number_of_classes: usize,
    pub(crate) criterion: SKConfiguredCriterion,
}

impl<F: SKFloat> SKClassifierCriterion<'_, F> {
    /// The weighted class counts of a node, in canonical class order.
    pub(crate) fn class_counts(&self, sample_indices: &[usize]) -> Vec<F> {
        let mut counts = vec![F::zero(); self.number_of_classes];
        for &row in sample_indices {
            counts[self.class_indices[row]] =
                counts[self.class_indices[row]] + self.sample_weights[row];
        }
        counts
    }
}

impl<F: SKFloat> SKTreeCriterion<F> for SKClassifierCriterion<'_, F> {
    fn impurity(&self, sample_indices: &[usize]) -> F {
        sk_classification_impurity(&self.class_counts(sample_indices), self.criterion)
    }

    fn leaf_output(&self, sample_indices: &[usize]) -> Vec<F> {
        let counts = self.class_counts(sample_indices);
        let total = counts.iter().fold(F::zero(), |sum, count| sum + *count);
        counts.iter().map(|count| *count / total).collect()
    }

    fn scan_context(&self) -> SKScanContext<'_, F> {
        SKScanContext::Classification {
            class_indices: self.class_indices,
            sample_weights: self.sample_weights,
            number_of_classes: self.number_of_classes,
            criterion: self.criterion,
        }
    }
}
