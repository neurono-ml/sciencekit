//! The reversible `SKLabelTable` mapping and its accessors.

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
    /// Build a table from an existing pair of maps. Used by canonicalization,
    /// which owns the only construction path.
    pub(crate) fn from_maps(
        label_to_index: HashMap<String, usize>,
        index_to_label: Vec<String>,
    ) -> Self {
        SKLabelTable {
            label_to_index,
            index_to_label,
        }
    }

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
