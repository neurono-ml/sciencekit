//! The CLI error type and its mapping to the stable exit codes.

use std::fmt;

use sciencekit::SKError;

use crate::cli::exit_codes::SKExitCode;

/// Every failure the CLI can report, classified by the exit code it maps to.
#[derive(Debug)]
pub enum SKCliError {
    /// A command, flag or configuration value is invalid.
    Validation {
        /// The dotted path of the offending value.
        path: String,
        /// Why the value is rejected.
        message: String,
    },
    /// The command surface itself was used incorrectly.
    Usage {
        /// Human-readable usage failure.
        message: String,
    },
    /// A dataset or model artifact could not be read or written.
    Data {
        /// The file path involved.
        path: String,
        /// What went wrong.
        message: String,
    },
    /// The build lacks the capability the command asked for.
    Unsupported {
        /// The Cargo feature that enables it.
        feature: &'static str,
        /// The command surface that requires it.
        surface: String,
    },
    /// An unexpected runtime failure.
    Runtime {
        /// What went wrong.
        message: String,
    },
}

impl SKCliError {
    /// Build a validation error for a dotted configuration or workflow path.
    pub fn validation(path: impl Into<String>, message: impl Into<String>) -> Self {
        SKCliError::Validation {
            path: path.into(),
            message: message.into(),
        }
    }

    /// Build a usage error.
    pub fn usage(message: impl Into<String>) -> Self {
        SKCliError::Usage {
            message: message.into(),
        }
    }

    /// Build a data I/O error.
    pub fn data(path: impl Into<String>, message: impl Into<String>) -> Self {
        SKCliError::Data {
            path: path.into(),
            message: message.into(),
        }
    }

    /// Build an unsupported-feature error.
    pub fn unsupported(feature: &'static str, surface: impl Into<String>) -> Self {
        SKCliError::Unsupported {
            feature,
            surface: surface.into(),
        }
    }

    /// Build a runtime error.
    pub fn runtime(message: impl Into<String>) -> Self {
        SKCliError::Runtime {
            message: message.into(),
        }
    }

    /// The documented exit code this failure maps to.
    pub const fn exit_code(&self) -> SKExitCode {
        match self {
            SKCliError::Validation { .. } | SKCliError::Usage { .. } => SKExitCode::Usage,
            SKCliError::Data { .. } => SKExitCode::Data,
            SKCliError::Unsupported { .. } => SKExitCode::UnsupportedFeature,
            SKCliError::Runtime { .. } => SKExitCode::Runtime,
        }
    }
}

impl fmt::Display for SKCliError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SKCliError::Validation { path, message } => {
                write!(formatter, "{path}: {message}")
            }
            SKCliError::Usage { message } => formatter.write_str(message),
            SKCliError::Data { path, message } => write!(formatter, "{path}: {message}"),
            SKCliError::Unsupported { feature, surface } => write!(
                formatter,
                "{surface} requires the `{feature}` Cargo feature, which this build was compiled without"
            ),
            SKCliError::Runtime { message } => formatter.write_str(message),
        }
    }
}

impl std::error::Error for SKCliError {}

/// Classify a library error raised while performing `operation`.
pub fn sk_classify_library_error(operation: &'static str, error: SKError) -> SKCliError {
    match error {
        SKError::Io(source) => SKCliError::data(operation, source.to_string()),
        SKError::UnsupportedOperation { .. } => {
            SKCliError::unsupported("sciencekit-operation", format!("{operation}: {error}"))
        }
        other => SKCliError::runtime(format!("{operation}: {other}")),
    }
}
