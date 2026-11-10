//! A single named, typed schema column (spec `dataset-interop-core`).

use super::{SKColumnRole, SKLogicalType};

/// The schema description of one column: its name, logical type and role.
///
/// A schema column carries no data. It is what an adapter declares before any
/// row is read, so selection and type validation happen up front.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SKColumnSchema {
    name: String,
    logical_type: SKLogicalType,
    role: SKColumnRole,
}

impl SKColumnSchema {
    /// Describe a column with the given name, logical type and role.
    pub fn new(
        name: impl Into<String>,
        logical_type: SKLogicalType,
        role: SKColumnRole,
    ) -> Self {
        SKColumnSchema {
            name: name.into(),
            logical_type,
            role,
        }
    }

    /// The column name used by name-based selection and in errors.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// The column's logical type.
    pub fn logical_type(&self) -> SKLogicalType {
        self.logical_type
    }

    /// The column's role.
    pub fn role(&self) -> SKColumnRole {
        self.role
    }

    /// Whether the column enters the numeric feature matrix directly.
    pub fn is_numeric(&self) -> bool {
        self.logical_type.is_numeric()
    }

    /// Whether the column enters the categorical view.
    pub fn is_categorical(&self) -> bool {
        self.logical_type.is_categorical()
    }
}
