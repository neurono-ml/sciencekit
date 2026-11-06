//! Classification label-space resolution (spec `evaluation-metrics`).

use std::collections::BTreeSet;

use sciencekit_common::SKError;

/// The resolved class labels of a classification metric.
///
/// An explicit list defines the order after validation; otherwise the space is
/// the sorted union of the observed true and predicted labels. Class order is
/// always ascending, independent of observation order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SKLabelSpace {
    labels: Vec<i64>,
}

impl SKLabelSpace {
    /// Resolve an optional explicit label list against the observed labels.
    pub fn resolve(
        explicit: Option<&[i64]>,
        true_labels: &[i64],
        predicted_labels: &[i64],
    ) -> Result<Self, SKError> {
        match explicit {
            Some(labels) => {
                let space = Self::from_explicit(labels)?;
                for label in true_labels.iter().chain(predicted_labels.iter()) {
                    space.index_of(*label)?;
                }
                Ok(space)
            }
            None => Ok(Self::from_observed(true_labels, predicted_labels)),
        }
    }

    /// Build the space from an explicit label list, rejecting empty and
    /// duplicated lists and sorting it internally.
    pub fn from_explicit(labels: &[i64]) -> Result<Self, SKError> {
        if labels.is_empty() {
            return Err(SKError::InvalidHyperparameter {
                name: "labels",
                reason: "at least one label is required".to_string(),
            });
        }
        let mut sorted = labels.to_vec();
        sorted.sort_unstable();
        for pair in sorted.windows(2) {
            if pair[0] == pair[1] {
                return Err(SKError::InvalidHyperparameter {
                    name: "labels",
                    reason: format!("label {} appears more than once", pair[0]),
                });
            }
        }
        Ok(Self { labels: sorted })
    }

    /// Build the sorted union of the observed true and predicted labels.
    pub fn from_observed(true_labels: &[i64], predicted_labels: &[i64]) -> Self {
        let labels = true_labels
            .iter()
            .chain(predicted_labels.iter())
            .copied()
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect();
        Self { labels }
    }

    /// The resolved labels in ascending order.
    pub fn labels(&self) -> &[i64] {
        &self.labels
    }

    /// The number of resolved labels.
    pub fn len(&self) -> usize {
        self.labels.len()
    }

    /// Whether the space holds no labels.
    pub fn is_empty(&self) -> bool {
        self.labels.is_empty()
    }

    /// The index of a label inside the resolved space.
    pub fn index_of(&self, label: i64) -> Result<usize, SKError> {
        self.labels
            .binary_search(&label)
            .map_err(|_| SKError::UnknownLabel { label })
    }
}
