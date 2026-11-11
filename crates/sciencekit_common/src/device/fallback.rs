//! The deterministic CPU fallback vocabulary (spec `gpu-compute-contracts`).
//!
//! When a device cannot serve an operation the plan falls back to this host
//! backend and records *why*, so the span tells the truth instead of silently
//! running on the CPU.

use std::fmt;

use crate::observability::SKBackendKind;

/// The host backend a plan falls back to when no device can serve it.
pub const SK_HOST_FALLBACK_BACKEND: SKBackendKind = SKBackendKind::Faer;

/// Why a plan did not select a device.
///
/// The reasons are bounded and render as stable kebab-case tokens recorded on
/// the operation span.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SKFallbackReason {
    /// No device of the requested backend kind is present in the inventory.
    NoDeviceAvailable,
    /// Devices are present, but none supports the kernel the operation needs.
    KernelNotSupported,
    /// Devices are present, but none supports the scalar type the operation carries.
    ScalarTypeNotSupported,
    /// The operation requested an explicit host execution mode, which never
    /// selects a device.
    ExplicitModeDoesNotSelectDevice,
}

impl fmt::Display for SKFallbackReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let token = match self {
            SKFallbackReason::NoDeviceAvailable => "no-device-available",
            SKFallbackReason::KernelNotSupported => "kernel-not-supported",
            SKFallbackReason::ScalarTypeNotSupported => "scalar-type-not-supported",
            SKFallbackReason::ExplicitModeDoesNotSelectDevice => {
                "explicit-mode-does-not-select-device"
            }
        };
        f.write_str(token)
    }
}
