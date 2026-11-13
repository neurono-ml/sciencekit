//! The FFI-facing error type that carries a stable status and a message.

use std::fmt;

use sciencekit_common::SKError;

use crate::array::SKFfiArrayError;
use crate::status::{SKFfiStatus, sk_ffi_status_for_error};

/// A failure crossing, or about to cross, the ABI boundary.
#[derive(Debug)]
pub enum SKFfiError {
    /// A descriptor failed validation.
    Array(SKFfiArrayError),
    /// A central library error.
    Central(SKError),
    /// The handle is null, unknown or of the wrong kind.
    InvalidHandle,
    /// An argument the ABI can diagnose itself.
    InvalidArgument(String),
    /// The configured family cannot perform the operation.
    Unsupported {
        /// The requested operation.
        operation: &'static str,
        /// Why the operation is unavailable.
        reason: String,
    },
    /// The operation requires a fitted model.
    NotFitted,
    /// A filesystem operation failed.
    Io(String),
    /// An unexpected internal failure, including a caught panic.
    Internal(String),
}

impl SKFfiError {
    /// The stable status this failure reports.
    pub fn status(&self) -> SKFfiStatus {
        match self {
            SKFfiError::Array(error) => error.status(),
            SKFfiError::Central(error) => sk_ffi_status_for_error(error),
            SKFfiError::InvalidHandle => SKFfiStatus::InvalidHandle,
            SKFfiError::NotFitted => SKFfiStatus::NotFitted,
            SKFfiError::Io(_) => SKFfiStatus::Io,
            SKFfiError::InvalidArgument(_) => SKFfiStatus::InvalidArgument,
            SKFfiError::Unsupported { .. } => SKFfiStatus::Unsupported,
            SKFfiError::Internal(_) => SKFfiStatus::Internal,
        }
    }

    /// Build an invalid-argument failure from anything displayable.
    pub fn invalid_argument(message: impl fmt::Display) -> SKFfiError {
        SKFfiError::InvalidArgument(message.to_string())
    }

    /// Rebuild an error from a bare status reported by the registry.
    pub fn from_status(status: SKFfiStatus) -> SKFfiError {
        match status {
            SKFfiStatus::InvalidHandle => SKFfiError::InvalidHandle,
            SKFfiStatus::NotFitted => SKFfiError::NotFitted,
            SKFfiStatus::Io => SKFfiError::Io("unknown I/O failure".to_string()),
            other => SKFfiError::Internal(format!("unexpected status `{}`", other.name())),
        }
    }
}

impl From<SKFfiArrayError> for SKFfiError {
    fn from(error: SKFfiArrayError) -> Self {
        SKFfiError::Array(error)
    }
}

impl From<SKError> for SKFfiError {
    fn from(error: SKError) -> Self {
        SKFfiError::Central(error)
    }
}

impl fmt::Display for SKFfiError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SKFfiError::Array(error) => write!(formatter, "{error}"),
            SKFfiError::Central(error) => write!(formatter, "{error}"),
            SKFfiError::InvalidHandle => {
                formatter.write_str("invalid handle: the pointer is null, unknown or already freed")
            }
            SKFfiError::InvalidArgument(message) => formatter.write_str(message),
            SKFfiError::Unsupported { operation, reason } => {
                write!(
                    formatter,
                    "operation `{operation}` is not supported: {reason}"
                )
            }
            SKFfiError::NotFitted => formatter.write_str("the model has not been fitted"),
            SKFfiError::Io(message) => write!(formatter, "I/O error: {message}"),
            SKFfiError::Internal(message) => write!(formatter, "internal error: {message}"),
        }
    }
}

impl std::error::Error for SKFfiError {}
