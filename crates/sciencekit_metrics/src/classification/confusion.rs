//! Confusion matrix over the resolved label space.

use sciencekit_common::SKError;

use super::label_space::SKLabelSpace;

/// The confusion matrix of a classification evaluation.
///
/// Rows are true labels, columns are predicted labels, and the cells hold
/// unsigned counts. The class order follows the resolved [`SKLabelSpace`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SKConfusionMatrix {
    labels: Vec<i64>,
    counts: Vec<usize>,
}

impl SKConfusionMatrix {
    /// The resolved labels in row and column order.
    pub fn labels(&self) -> &[i64] {
        &self.labels
    }

    /// The row-major counts, `classes × classes` entries.
    pub fn counts(&self) -> &[usize] {
        &self.counts
    }

    /// The matrix dimensions, both equal to the number of resolved labels.
    pub fn shape(&self) -> (usize, usize) {
        (self.labels.len(), self.labels.len())
    }

    /// The number of observations counted by the matrix.
    pub fn number_of_observations(&self) -> usize {
        self.counts.iter().sum()
    }

    /// The count at a true/predicted label pair, or `None` for an unknown label.
    pub fn get(&self, true_label: i64, predicted_label: i64) -> Option<usize> {
        let true_index = self.labels.binary_search(&true_label).ok()?;
        let predicted_index = self.labels.binary_search(&predicted_label).ok()?;
        Some(self.at(true_index, predicted_index))
    }

    /// The number of resolved classes.
    pub(crate) fn number_of_classes(&self) -> usize {
        self.labels.len()
    }

    /// The count at a true/predicted class index.
    pub(crate) fn at(&self, true_index: usize, predicted_index: usize) -> usize {
        self.counts[true_index * self.labels.len() + predicted_index]
    }

    /// The true positives of one class.
    pub(crate) fn true_positive(&self, index: usize) -> usize {
        self.at(index, index)
    }

    /// The number of observations whose true label is the class at `index`.
    pub(crate) fn support(&self, index: usize) -> usize {
        (0..self.labels.len()).map(|column| self.at(index, column)).sum()
    }

    /// The number of predictions of the class at `index`.
    pub(crate) fn prediction_count(&self, index: usize) -> usize {
        (0..self.labels.len()).map(|row| self.at(row, index)).sum()
    }
}

/// Build the confusion matrix over the resolved label space.
///
/// An explicit label list fixes the class order; otherwise the space is the
/// sorted union of the observed labels.
pub fn sk_confusion_matrix(
    true_labels: &[i64],
    predicted_labels: &[i64],
    explicit_labels: Option<&[i64]>,
) -> Result<SKConfusionMatrix, SKError> {
    if true_labels.is_empty() {
        return Err(SKError::EmptyInput {
            operation: "confusion_matrix",
        });
    }
    if true_labels.len() != predicted_labels.len() {
        return Err(SKError::shape_mismatch_2d(
            true_labels.len(),
            1,
            predicted_labels.len(),
            1,
        ));
    }
    let space = SKLabelSpace::resolve(explicit_labels, true_labels, predicted_labels)?;
    let classes = space.len();
    let mut counts = vec![0_usize; classes * classes];
    for (true_label, predicted_label) in true_labels.iter().zip(predicted_labels.iter()) {
        let true_index = space.index_of(*true_label)?;
        let predicted_index = space.index_of(*predicted_label)?;
        counts[true_index * classes + predicted_index] += 1;
    }
    Ok(SKConfusionMatrix {
        labels: space.labels().to_vec(),
        counts,
    })
}
