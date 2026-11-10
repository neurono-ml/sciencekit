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
    ) {
        Ok(selected) => (selected.kind, None),
        Err(reason) => (SK_HOST_FALLBACK_BACKEND, Some(reason)),
    }
}
