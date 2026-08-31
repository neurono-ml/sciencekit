//! Canonical label tables (spec `data-view-boundary`).
//!
//! Deterministic canonicalization of nominal label sequences into compact
//! indices plus a reversible table — the foundation of classifier automatic
//! encoding (Phase 1+) and of explicit codecs.

use std::collections::HashMap;

/// A reversible label↔index mapping produced by canonicalization.
///
/// The table is the bearer of exportable metadata (it feeds the future model
/// header, PRD §8.2): it exposes readable label↔index access.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SKLabelTable {
    label_to_index: HashMap<String, usize>,
    index_to_label: Vec<String>,
}

impl SKLabelTable {
    /// The compact index for a label, if present.
    pub fn index_of(&self, label: &str) -> Option<usize> {
        self.label_to_index.get(label).copied()
    }

    /// The label for a compact index, if present.
    pub fn label_of(&self, index: usize) -> Option<&str> {
        self.index_to_label.get(index).map(String::as_str)
    }

    /// The number of distinct classes.
    pub fn number_of_classes(&self) -> usize {
        self.index_to_label.len()
    }

    /// Iterate the labels in index order.
    pub fn labels(&self) -> impl Iterator<Item = &str> {
        self.index_to_label.iter().map(String::as_str)
    }
}

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
        SKLabelTable {
            label_to_index,
            index_to_label,
        },
    )
}

#[cfg(test)]
mod label_table_tests;
