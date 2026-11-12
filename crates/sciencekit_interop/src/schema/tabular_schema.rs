//! The ordered schema of a tabular dataset (spec `dataset-interop-core`).

use std::collections::HashSet;

use sciencekit_common::SKError;

use super::{SKColumnRole, SKColumnSchema};

/// An ordered, validated description of a table's columns.
///
/// Construction enforces the two invariants the whole seam relies on: the
/// schema is non-empty, and every column name is unique. A duplicate name is
/// rejected here, before any row is read, because name-based selection would
/// otherwise be ambiguous.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SKTabularSchema {
    columns: Vec<SKColumnSchema>,
}

impl SKTabularSchema {
    /// Build a schema from columns in source order.
    ///
    /// # Errors
    /// Returns [`SKError::SchemaMismatch`] for an empty schema and
    /// [`SKError::DuplicateColumn`] when two columns share a name.
    pub fn new(columns: Vec<SKColumnSchema>) -> Result<Self, SKError> {
        if columns.is_empty() {
            return Err(SKError::SchemaMismatch {
                expected: "at least one column".to_string(),
                found: "a schema with zero columns".to_string(),
            });
        }

        let mut seen_names = HashSet::with_capacity(columns.len());
        for column in &columns {
            if !seen_names.insert(column.name()) {
                return Err(SKError::DuplicateColumn {
                    column: column.name().to_string(),
                    suggestion: "rename one of the duplicate columns at the source".to_string(),
                });
            }
        }

        Ok(SKTabularSchema { columns })
    }

    /// The columns in source order.
    pub fn columns(&self) -> &[SKColumnSchema] {
        &self.columns
    }

    /// The number of columns.
    pub fn len(&self) -> usize {
        self.columns.len()
    }

    /// Whether the schema carries no column (never true for a constructed
    /// schema, kept for symmetry with the standard container API).
    pub fn is_empty(&self) -> bool {
        self.columns.is_empty()
    }

    /// The column at a zero-based position.
    pub fn column_at(&self, position: usize) -> Option<&SKColumnSchema> {
        self.columns.get(position)
    }

    /// The zero-based position of a column name.
    ///
    /// Lookup is deterministic by construction: names are unique, so the first
    /// (and only) match is the answer.
    pub fn column_index(&self, name: &str) -> Option<usize> {
        self.columns.iter().position(|column| column.name() == name)
    }

    /// The column with the given name.
    pub fn column(&self, name: &str) -> Option<&SKColumnSchema> {
        self.column_index(name).map(|index| &self.columns[index])
    }

    /// The zero-based positions of every column carrying a role.
    pub fn column_indices_with_role(&self, role: SKColumnRole) -> Vec<usize> {
        self.columns
            .iter()
            .enumerate()
            .filter(|(_, column)| column.role() == role)
            .map(|(index, _)| index)
            .collect()
    }
}
