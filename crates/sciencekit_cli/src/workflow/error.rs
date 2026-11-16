//! Typed workflow validation errors; each one names the offending JSON path.

use std::fmt;

use crate::error::SKCliError;

/// A workflow document that cannot be accepted.
#[derive(Debug, PartialEq, Eq)]
pub enum SKWorkflowError {
    /// The document declares a major version this build does not implement.
    UnsupportedVersion {
        /// The dotted path of the version field.
        path: String,
        /// The version found in the document.
        found: u64,
        /// The version this build implements.
        supported: u32,
    },
    /// The document carries a key the schema does not define.
    UnknownField {
        /// The dotted path of the offending key.
        path: String,
        /// The key itself.
        field: String,
        /// The keys accepted at that path.
        expected: Vec<&'static str>,
    },
    /// A required key is absent.
    MissingField {
        /// The dotted path of the missing key.
        path: String,
    },
    /// A key carries a value of the wrong JSON type.
    InvalidType {
        /// The dotted path of the offending key.
        path: String,
        /// The JSON type the schema requires.
        expected: &'static str,
    },
}

impl SKWorkflowError {
    /// The dotted path the error points at.
    pub fn path(&self) -> &str {
        match self {
            SKWorkflowError::UnsupportedVersion { path, .. }
            | SKWorkflowError::UnknownField { path, .. }
            | SKWorkflowError::MissingField { path }
            | SKWorkflowError::InvalidType { path, .. } => path,
        }
    }

    /// Convert into the CLI's validation error, preserving the path.
    pub fn into_cli_error(self) -> SKCliError {
        let path = self.path().to_string();
        SKCliError::validation(path, self.to_string())
    }
}

impl fmt::Display for SKWorkflowError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SKWorkflowError::UnsupportedVersion {
                path,
                found,
                supported,
            } => write!(
                formatter,
                "{path} declares workflow version {found}; this build supports version {supported}"
            ),
            SKWorkflowError::UnknownField {
                path,
                field,
                expected,
            } => write!(
                formatter,
                "{path} carries the unknown field `{field}`; expected one of: {}",
                expected.join(", ")
            ),
            SKWorkflowError::MissingField { path } => {
                write!(formatter, "{path} is required but absent")
            }
            SKWorkflowError::InvalidType { path, expected } => {
                write!(formatter, "{path} must be a {expected}")
            }
        }
    }
}

impl std::error::Error for SKWorkflowError {}
