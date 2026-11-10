//! A reference to a schema column by name or position (spec
//! `dataset-interop-core`).

use sciencekit_common::SKError;

use crate::schema::SKTabularSchema;

/// How a selection names a column: by its schema name or by its position.
///
/// Positional selection exists for sources whose column names are not stable or
/// are not unique; name-based selection is the readable default.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum SKColumnReference {
    /// Select by column name.
    Name(String),
    /// Select by zero-based column position.
    Position(usize),
}

impl SKColumnReference {
    /// A reference to a column by name.
    pub fn name(name: impl Into<String>) -> Self {
        SKColumnReference::Name(name.into())
    }

    /// A reference to a column by zero-based position.
    pub fn position(position: usize) -> Self {
        SKColumnReference::Position(position)
    }

    /// Resolve the reference against a schema, returning the column position.
    ///
    /// # Errors
    /// Returns [`SKError::UnknownColumn`] naming the missing column when the
    /// name or position is not present in the schema.
    pub fn resolve(&self, schema: &SKTabularSchema) -> Result<usize, SKError> {
        match self {
            SKColumnReference::Name(name) => {
                schema
                    .column_index(name)
                    .ok_or_else(|| SKError::UnknownColumn {
                        column: name.clone(),
                        suggestion: "check the schema for the exact column name".to_string(),
                    })
            }
            SKColumnReference::Position(position) => {
                if *position < schema.len() {
                    Ok(*position)
                } else {
                    Err(SKError::UnknownColumn {
                        column: format!("position {position}"),
                        suggestion: format!(
                            "the schema has {} columns, so positions run from 0 to {}",
                            schema.len(),
                            schema.len().saturating_sub(1)
                        ),
                    })
                }
            }
        }
    }
}
