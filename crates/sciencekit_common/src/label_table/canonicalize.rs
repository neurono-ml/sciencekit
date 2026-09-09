//! Deterministic canonicalization of nominal labels into indices.

use std::collections::HashMap;

use super::SKLabelTable;

/// Canonicalize a sequence of labels into compact indices and a reversible
/// table. Deterministic: the same input always yields the same mapping
/// (indices assigned in first-occurrence order).
pub fn sk_canonicalize_labels<'a>(labels: &'a [&'a str]) -> (Vec<usize>, SKLabelTable) {
    let mut label_to_index = HashMap::new();
    let mut index_to_label: Vec<String> = Vec::new();
    let mut indices = Vec::with_capacity(labels.len());

    for label in labels {
        let index = *label_to_index
            .entry((*label).to_owned())
            .or_insert_with(|| {
                index_to_label.push((*label).to_owned());
                index_to_label.len() - 1
            });
        indices.push(index);
    }

    (
        indices,
        SKLabelTable::from_maps(label_to_index, index_to_label),
    )
}
