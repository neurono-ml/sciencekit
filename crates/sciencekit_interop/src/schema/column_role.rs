//! The role a schema column plays in a dataset (spec `dataset-interop-core`).

use std::fmt;
use std::str::FromStr;

use sciencekit_common::SKError;

/// The role assigned to a column: a feature, the target, or ignored.
///
/// Roles document intent at the schema level. The authoritative feature order
/// is the declared selection order, never the source's incidental order; the
/// role is what lets an adapter describe a table before any row is read.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SKColumnRole {
    /// The column contributes to the feature matrix.
    Feature,
    /// The column carries the response of a supervised problem.
    Target,
    /// The column is present in the source but does not enter the model.
    Ignored,
}

impl SKColumnRole {
    /// The stable, lower-case name used in errors and metadata.
    pub fn name(&self) -> &'static str {
        match self {
            SKColumnRole::Feature => "feature",
            SKColumnRole::Target => "target",
            SKColumnRole::Ignored => "ignored",
        }
    }
}

impl fmt::Display for SKColumnRole {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.name())
    }
}

impl FromStr for SKColumnRole {
    type Err = SKError;

    fn from_str(name: &str) -> Result<Self, Self::Err> {
        match name {
            "feature" => Ok(SKColumnRole::Feature),
            "target" => Ok(SKColumnRole::Target),
            "ignored" => Ok(SKColumnRole::Ignored),
            other => Err(SKError::Conversion(format!(
                "unknown column role `{other}`; expected one of feature, target, ignored"
            ))),
        }
    }
}
