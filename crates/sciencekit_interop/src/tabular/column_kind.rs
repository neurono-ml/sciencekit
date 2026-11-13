//! The kind a tabular source assigns to a column (spec `tabular-sources`).

use crate::schema::SKLogicalType;

/// How a column enters a [`super::SKTabularDataset`].
///
/// The kind is the caller's declaration; it is validated against the column's
/// logical type before any row is materialized, so an accidental mismatch is a
/// structured error rather than a silently wrong matrix.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SKColumnKind {
    /// A numeric column, including integers promoted to the model scalar.
    Numeric,
    /// A nominal or category-coded column.
    Categorical,
    /// A boolean column encoded as `0`/`1` on the numeric path.
    Boolean,
    /// A column carried by the source but excluded from the outputs.
    Ignored,
}

impl SKColumnKind {
    /// The stable, lower-case name used in errors and traces.
    pub fn name(&self) -> &'static str {
        match self {
            SKColumnKind::Numeric => "numeric",
            SKColumnKind::Categorical => "categorical",
            SKColumnKind::Boolean => "boolean",
            SKColumnKind::Ignored => "ignored",
        }
    }

    /// Whether the kind produces a numeric feature column.
    pub fn is_numeric(&self) -> bool {
        matches!(self, SKColumnKind::Numeric | SKColumnKind::Boolean)
    }

    /// Whether the kind feeds the categorical block.
    pub fn is_categorical(&self) -> bool {
        matches!(self, SKColumnKind::Categorical)
    }

    /// Whether a logical type may carry this kind.
    pub fn accepts(&self, logical_type: SKLogicalType) -> bool {
        match self {
            SKColumnKind::Numeric => logical_type.is_numeric(),
            SKColumnKind::Boolean => logical_type == SKLogicalType::Boolean,
            SKColumnKind::Categorical => logical_type.is_categorical(),
            SKColumnKind::Ignored => true,
        }
    }
}

impl std::fmt::Display for SKColumnKind {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.name())
    }
}
