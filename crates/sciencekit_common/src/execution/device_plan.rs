//! Device selection for execution planning (spec `gpu-compute-contracts`).
//!
//! The decision is a pure function of the intent and the context's device
//! request: an automatic plan may select the configured device when the
//! inventory carries one that implements the required kernel; every other case
//! plans on a host backend and records why the device was not used.

use crate::device::{
    SK_HOST_FALLBACK_BACKEND, SKDeviceContext, SKDevicePreference, SKFallbackReason,
    sk_select_device,
};
use crate::observability::SKBackendKind;

use super::modes::SKExecutionMode;

/// The backend a plan runs on, plus the reason a device was not selected.
pub(crate) fn plan_backend(
    intent: SKExecutionMode,
    device: Option<&SKDeviceContext>,
    scalar_size_bytes: u64,
) -> (SKBackendKind, Option<SKFallbackReason>) {
    let Some(device) = device else {
        return (SKBackendKind::default(), None);
    };
    let configured = device.configured_backend;
    if configured.is_host() {
        return (configured, None);
    }
    if intent != SKExecutionMode::Automatic {
        return (
            SK_HOST_FALLBACK_BACKEND,
            Some(SKFallbackReason::ExplicitModeDoesNotSelectDevice),
        );
    }
    match sk_select_device(
        &device.device_inventory,
        SKDevicePreference::Backend(configured),
        device.required_kernel,
        scalar_type_for_size(scalar_size_bytes),
    ) {
        Ok(selected) => (selected.kind, None),
        Err(reason) => (SK_HOST_FALLBACK_BACKEND, Some(reason)),
    }
}

/// Map the context's scalar byte width onto the capability name (`"f32"` or `"f64"`).
fn scalar_type_for_size(scalar_size_bytes: u64) -> Option<&'static str> {
    match scalar_size_bytes {
        0 => None,
        4 => Some("f32"),
        8 => Some("f64"),
        _ => Some("unsupported"),
    }
}
