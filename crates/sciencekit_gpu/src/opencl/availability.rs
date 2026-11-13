//! Availability of the OpenCL implementation behind the `opencl` feature.
//!
//! The fallback must be honest: every operation that cannot reach a device
//! reports *why*. This module owns the two answers the rest of the module asks
//! — is the feature compiled in, and what reason describes the unavailability —
//! so no call site hardcodes a misleading string.

use sciencekit_common::{SKBackendKind, SKError, SKFallbackReason};

/// Whether the `opencl` feature was compiled in.
///
/// With the feature off no OpenCL symbol is ever called; the backend still
/// exists so planning can record the CPU fallback instead of failing to build
/// the call site.
pub(crate) fn is_feature_enabled() -> bool {
    cfg!(feature = "opencl")
}

/// The reason recorded when the backend cannot serve an operation.
pub(crate) fn fallback_reason() -> SKFallbackReason {
    SKFallbackReason::NoDeviceAvailable
}

/// The structured error describing why no device can serve an operation.
pub(crate) fn unavailable_error(reason: impl Into<String>) -> SKError {
    SKError::DeviceUnavailable {
        backend: SKBackendKind::OpenCl.name(),
        reason: reason.into(),
    }
}

/// The reason string used when the feature is off.
pub(crate) fn feature_disabled_reason() -> String {
    "the `opencl` feature of `sciencekit_gpu` is disabled; CPU execution runs instead".to_string()
}
