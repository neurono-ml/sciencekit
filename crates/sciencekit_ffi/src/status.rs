//! Integer status codes returned by every fallible entry point (spec
//! `ffi-abi`).

use sciencekit_common::SKError;

/// Stable status codes shared with C.
///
/// The discriminants are part of the ABI: never renumber them. New failures
/// append a variant and bump the minor version.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SKFfiStatus {
    /// The operation completed successfully; the last-error message is empty.
    Ok = 0,
    /// The handle is null, unknown or of the wrong kind.
    InvalidHandle = 1,
    /// The shapes or lengths of the inputs are incompatible.
    Shape = 2,
    /// An argument is malformed (descriptor, dtype, path, output pointer).
    InvalidArgument = 3,
    /// The operation or representation is not supported by this configuration.
    Unsupported = 4,
    /// The operation requires a fitted model.
    NotFitted = 5,
    /// A filesystem operation failed.
    Io = 6,
    /// An unexpected internal failure, including a caught panic.
    Internal = 7,
}

impl SKFfiStatus {
    /// The stable lowercase name a binding can display.
    pub fn name(self) -> &'static str {
        match self {
            SKFfiStatus::Ok => "ok",
            SKFfiStatus::InvalidHandle => "invalid-handle",
            SKFfiStatus::Shape => "shape",
            SKFfiStatus::InvalidArgument => "invalid-argument",
            SKFfiStatus::Unsupported => "unsupported",
            SKFfiStatus::NotFitted => "not-fitted",
            SKFfiStatus::Io => "io",
            SKFfiStatus::Internal => "internal",
        }
    }

    /// Resolve a code received from a binding, rejecting unknown values.
    pub fn from_code(code: i32) -> Option<SKFfiStatus> {
        match code {
            0 => Some(SKFfiStatus::Ok),
            1 => Some(SKFfiStatus::InvalidHandle),
            2 => Some(SKFfiStatus::Shape),
            3 => Some(SKFfiStatus::InvalidArgument),
            4 => Some(SKFfiStatus::Unsupported),
            5 => Some(SKFfiStatus::NotFitted),
            6 => Some(SKFfiStatus::Io),
            7 => Some(SKFfiStatus::Internal),
            _ => None,
        }
    }
}

/// Map a central `SKError` onto its stable ABI status.
pub fn sk_ffi_status_for_error(error: &SKError) -> SKFfiStatus {
    match error {
        SKError::ShapeMismatch { .. } => SKFfiStatus::Shape,
        SKError::Io(_) => SKFfiStatus::Io,
        SKError::UnsupportedRepresentation { .. }
        | SKError::UnsupportedOperation { .. }
        | SKError::ExecutionModeIncompatible { .. }
        | SKError::BatchBufferOversize { .. } => SKFfiStatus::Unsupported,
        SKError::InvalidHyperparameter { .. }
        | SKError::UnexpectedMissingness { .. }
        | SKError::Conversion(_)
        | SKError::EmptyInput { .. }
        | SKError::NonFiniteValue { .. }
        | SKError::ProbabilityOutOfRange { .. }
        | SKError::UnknownLabel { .. }
        | SKError::UnknownColumn { .. }
        | SKError::DuplicateColumn { .. }
        | SKError::UnsupportedColumnType { .. }
        | SKError::NullValueInColumn { .. }
        | SKError::UnevenColumnLengths { .. }
        | SKError::SchemaMismatch { .. } => SKFfiStatus::InvalidArgument,
        SKError::NotConverged { .. } => SKFfiStatus::Internal,
        _ => SKFfiStatus::Internal,
    }
}

#[cfg(test)]
#[path = "status_tests.rs"]
mod status_tests;
