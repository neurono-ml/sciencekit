//! Structured errors and their mapping onto Lua errors (spec `lua-bindings`).
//!
//! Every failure carries a stable [`SKFfiStatus`] code and the message the C
//! ABI recorded. The Lua error string names both, so Lua code can branch on
//! the status family without parsing free text:
//!
//! ```text
//! sciencekit shape (2): features must be a rectangular table of rows
//! ```

use std::fmt;

use sciencekit_ffi::SKFfiStatus;

/// A binding failure: a status code plus a human-readable message.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct SKLuaError {
    status: SKFfiStatus,
    message: String,
}

impl SKLuaError {
    /// Build an error with an explicit status and message.
    pub(crate) fn new(status: SKFfiStatus, message: impl Into<String>) -> SKLuaError {
        SKLuaError {
            status,
            message: message.into(),
        }
    }

    /// Build an error from the status a failed ABI call returned, copying the
    /// thread-local last-error message the ABI recorded.
    pub(crate) fn from_last_error(status: SKFfiStatus) -> SKLuaError {
        let message = sciencekit_ffi::error::sk_ffi_last_error_message();
        let message = if message.is_empty() {
            status.name().to_string()
        } else {
            message
        };
        SKLuaError { status, message }
    }

    /// A shape error raised by table conversion before any ABI call.
    pub(crate) fn shape(message: impl Into<String>) -> SKLuaError {
        SKLuaError::new(SKFfiStatus::Shape, message)
    }

    /// A malformed argument raised by table conversion before any ABI call.
    pub(crate) fn invalid_argument(message: impl Into<String>) -> SKLuaError {
        SKLuaError::new(SKFfiStatus::InvalidArgument, message)
    }

    /// An operation attempted on a closed or freed handle.
    pub(crate) fn invalid_handle(message: impl Into<String>) -> SKLuaError {
        SKLuaError::new(SKFfiStatus::InvalidHandle, message)
    }

    /// An operation a family does not support.
    pub(crate) fn unsupported(message: impl Into<String>) -> SKLuaError {
        SKLuaError::new(SKFfiStatus::Unsupported, message)
    }

    /// The stable status name, for assertions in tests.
    #[cfg(test)]
    pub(crate) fn status_name(&self) -> &'static str {
        self.status.name()
    }

    /// The status discriminant carried in the message.
    pub(crate) fn status_code(&self) -> i32 {
        self.status as i32
    }

    /// The human-readable message without the status prefix.
    #[cfg(test)]
    pub(crate) fn message(&self) -> &str {
        &self.message
    }
}

impl fmt::Display for SKLuaError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "sciencekit {} ({}): {}",
            self.status.name(),
            self.status_code(),
            self.message
        )
    }
}

impl std::error::Error for SKLuaError {}

impl From<SKLuaError> for mlua::Error {
    fn from(error: SKLuaError) -> mlua::Error {
        mlua::Error::RuntimeError(error.to_string())
    }
}

impl From<mlua::Error> for SKLuaError {
    fn from(error: mlua::Error) -> SKLuaError {
        SKLuaError::invalid_argument(error.to_string())
    }
}
