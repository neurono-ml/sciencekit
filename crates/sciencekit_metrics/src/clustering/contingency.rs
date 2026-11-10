//! Contingency tables shared by the pairwise labeling metrics.

use std::collections::BTreeMap;

use ndarray::ArrayView1;
use sciencekit_common::SKError;

/// A sparse contingency table over the union label space.
///
/// Only non-empty cells are stored, so the memory is `O(number of rows)` even
/// when the label space is large. Labels are canonicalized through a
/// `BTreeMap`, which makes the metric independent of the numeric label values.
pub(super) struct SKContingencyTable {
    /// Number of paired observations.
    pub number_of_rows: usize,
    /// `(true index, predicted index, count)` for every non-empty cell.
    pub cells: Vec<(usize, usize, usize)>,
    /// Row sums over the true labels.
    pub true_counts: Vec<usize>,
    /// Row sums over the predicted labels.
    pub predicted_counts: Vec<usize>,
}

/// Build the contingency table from two label vectors.
pub(super) fn sk_contingency_table(
    labels_true: ArrayView1<'_, i64>,
    labels_pred: ArrayView1<'_, i64>,
    operation: &'static str,
) -> Result<SKContingencyTable, SKError> {
    if labels_true.is_empty() {
        return Err(SKError::EmptyInput { operation });
    }
    if labels_true.len() != labels_pred.len() {
        return Err(SKError::ShapeMismatch {
            expected: vec![labels_true.len()],
            found: vec![labels_pred.len()],
        });
    }

    let mut true_indices = BTreeMap::new();
    let mut predicted_indices = BTreeMap::new();
    let mut cells: BTreeMap<(usize, usize), usize> = BTreeMap::new();
    for (true_label, predicted_label) in labels_true.iter().zip(labels_pred.iter()) {
        let number_of_true_labels = true_indices.len();
        let true_index = *true_indices
            .entry(*true_label)
            .or_insert(number_of_true_labels);
        let number_of_predicted_labels = predicted_indices.len();
        let predicted_index = *predicted_indices
            .entry(*predicted_label)
            .or_insert(number_of_predicted_labels);
        *cells.entry((true_index, predicted_index)).or_insert(0) += 1;
    }

    let mut true_counts = vec![0; true_indices.len()];
    let mut predicted_counts = vec![0; predicted_indices.len()];
    for ((true_index, predicted_index), count) in cells.iter() {
        true_counts[*true_index] += count;
        predicted_counts[*predicted_index] += count;
    }
    let cells = cells
        .into_iter()
        .map(|((true_index, predicted_index), count)| (true_index, predicted_index, count))
        .collect();

    Ok(SKContingencyTable {
        number_of_rows: labels_true.len(),
        cells,
        true_counts,
        predicted_counts,
    })
}

/// Number of unordered pairs in a group of `count` rows.
pub(super) fn sk_pair_count(count: usize) -> f64 {
    let count = count as f64;
    count * (count - 1.0) / 2.0
}
