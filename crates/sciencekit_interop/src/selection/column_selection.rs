//! Explicit feature and target selection (spec `dataset-interop-core`).

use sciencekit_common::SKError;

use super::SKColumnReference;
use crate::schema::SKTabularSchema;

/// An ordered feature selection plus at most one target column.
///
/// The order of [`SKColumnSelection::feature_columns`] is the feature order of
/// every materialized block, regardless of the source's incidental column
/// order. Construction validates the selection surface; resolution against a
/// schema validates names, positions and duplicates before any row is read.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct SKColumnSelection {
    feature_columns: Vec<SKColumnReference>,
    target_columns: Vec<SKColumnReference>,
}

impl SKColumnSelection {
    /// Start building a selection.
    pub fn builder() -> SKColumnSelectionBuilder {
        SKColumnSelectionBuilder::default()
    }

    /// The feature references in declared order.
    pub fn feature_columns(&self) -> &[SKColumnReference] {
        &self.feature_columns
    }

    /// The declared target reference, when the selection is supervised.
    pub fn target_column(&self) -> Option<&SKColumnReference> {
        self.target_columns.first()
    }

    /// Whether the selection declares a target column.
    pub fn has_target(&self) -> bool {
        !self.target_columns.is_empty()
    }

    /// Resolve every reference against a schema.
    ///
    /// # Errors
    /// Returns a structured error for an unknown column, a duplicate feature
    /// selection, a target also selected as a feature, or an empty selection.
    pub fn resolve(&self, schema: &SKTabularSchema) -> Result<SKResolvedSelection, SKError> {
        if self.feature_columns.is_empty() {
            return Err(SKError::InvalidHyperparameter {
                name: "feature_columns",
                reason: "a column selection must name at least one feature column".to_string(),
            });
        }

        let mut feature_indices = Vec::with_capacity(self.feature_columns.len());
        for reference in &self.feature_columns {
            let index = reference.resolve(schema)?;
            if feature_indices.contains(&index) {
                return Err(SKError::DuplicateColumn {
                    column: schema.columns()[index].name().to_string(),
                    suggestion: "remove the repeated feature reference".to_string(),
                });
            }
            feature_indices.push(index);
        }

        let target_index = match self.target_columns.first() {
            Some(reference) => {
                let index = reference.resolve(schema)?;
                if feature_indices.contains(&index) {
                    return Err(SKError::DuplicateColumn {
                        column: schema.columns()[index].name().to_string(),
                        suggestion:
                            "remove the target column from the feature list or choose another target"
                                .to_string(),
                    });
                }
                Some(index)
            }
            None => None,
        };

        Ok(SKResolvedSelection {
            feature_indices,
            target_index,
        })
    }
}

/// A selection builder that validates its surface on completion.
#[derive(Debug, Clone, Default)]
pub struct SKColumnSelectionBuilder {
    feature_columns: Vec<SKColumnReference>,
    target_columns: Vec<SKColumnReference>,
}

impl SKColumnSelectionBuilder {
    /// Append a feature column by name; order is preserved.
    pub fn feature(mut self, name: impl Into<String>) -> Self {
        self.feature_columns.push(SKColumnReference::name(name));
        self
    }

    /// Append a feature column by zero-based position; order is preserved.
    pub fn feature_at(mut self, position: usize) -> Self {
        self.feature_columns
            .push(SKColumnReference::position(position));
        self
    }

    /// Declare the target column by name.
    pub fn target(mut self, name: impl Into<String>) -> Self {
        self.target_columns.push(SKColumnReference::name(name));
        self
    }

    /// Declare the target column by zero-based position.
    pub fn target_at(mut self, position: usize) -> Self {
        self.target_columns
            .push(SKColumnReference::position(position));
        self
    }

    /// Finish the selection.
    ///
    /// # Errors
    /// Returns [`SKError::InvalidHyperparameter`] for an empty feature list and
    /// [`SKError::UnsupportedOperation`] when more than one target is declared.
    pub fn build(self) -> Result<SKColumnSelection, SKError> {
        if self.feature_columns.is_empty() {
            return Err(SKError::InvalidHyperparameter {
                name: "feature_columns",
                reason: "a column selection must name at least one feature column".to_string(),
            });
        }
        if self.target_columns.len() > 1 {
            return Err(SKError::UnsupportedOperation {
                operation: "column_selection",
                reason: "at most one target column is supported".to_string(),
            });
        }
        Ok(SKColumnSelection {
            feature_columns: self.feature_columns,
            target_columns: self.target_columns,
        })
    }
}

/// A selection resolved against a concrete schema.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SKResolvedSelection {
    feature_indices: Vec<usize>,
    target_index: Option<usize>,
}

impl SKResolvedSelection {
    /// The feature column positions in selection order.
    pub fn feature_indices(&self) -> &[usize] {
        &self.feature_indices
    }

    /// The target column position, when supervised.
    pub fn target_index(&self) -> Option<usize> {
        self.target_index
    }

    /// The number of selected features.
    pub fn number_of_features(&self) -> usize {
        self.feature_indices.len()
    }
}
