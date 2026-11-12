//! Process-wide device inventory populated explicitly by device runtimes
//! (spec `gpu-compute-contracts`).
//!
//! Nothing is guessed or probed automatically: the registry starts empty so
//! default CPU builds incur zero runtime probing and report honest CPU
//! fallbacks whenever a device backend kind is requested without a registered
//! device. A concrete runtime (`sciencekit_gpu_opencl`, `sciencekit_gpu_cuda`,
//! `sciencekit_gpu_rocm`) registers the devices it enumerated.

use std::sync::RwLock;

use crate::observability::SKBackendKind;

use super::fallback::{SK_HOST_FALLBACK_BACKEND, SKFallbackReason};
use super::info::SKDeviceInfo;
use super::selection::{SKDevicePreference, sk_select_device};

static REGISTERED_DEVICE_INVENTORY: RwLock<Vec<SKDeviceInfo>> = RwLock::new(Vec::new());

/// Replace the process-wide device inventory with `inventory`.
pub fn sk_register_device_inventory(inventory: Vec<SKDeviceInfo>) {
    let mut guard = REGISTERED_DEVICE_INVENTORY
        .write()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    *guard = inventory;
}

/// Snapshot the currently registered process-wide device inventory.
pub fn sk_registered_device_inventory() -> Vec<SKDeviceInfo> {
    let guard = REGISTERED_DEVICE_INVENTORY
        .read()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    guard.clone()
}

/// Clear every device from the process-wide inventory.
pub fn sk_clear_device_inventory() {
    let mut guard = REGISTERED_DEVICE_INVENTORY
        .write()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    guard.clear();
}

/// Resolve the backend that actually serves an operation configured with
/// `configured`, consulting the registered device inventory for device kinds.
///
/// A host kind resolves to itself with no fallback reason. A device kind
/// resolves through [`sk_select_device`] over the registered inventory with
/// [`SKDevicePreference::Backend`], falling back to
/// [`SK_HOST_FALLBACK_BACKEND`] and returning the [`SKFallbackReason`] when no
/// compatible device is registered.
pub fn sk_resolve_operation_backend(
    configured: SKBackendKind,
) -> (SKBackendKind, Option<SKFallbackReason>) {
    if configured.is_host() {
        return (configured, None);
    }
    let guard = REGISTERED_DEVICE_INVENTORY
        .read()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    match sk_select_device(&guard, SKDevicePreference::Backend(configured), None, None) {
        Ok(selected) => (selected.kind, None),
        Err(reason) => (SK_HOST_FALLBACK_BACKEND, Some(reason)),
    }
}

#[cfg(test)]
pub(crate) struct DeviceInventoryTestGuard {
    _guard: std::sync::MutexGuard<'static, ()>,
}

#[cfg(test)]
impl Drop for DeviceInventoryTestGuard {
    fn drop(&mut self) {
        sk_clear_device_inventory();
    }
}

#[cfg(test)]
pub(crate) fn lock_device_inventory_for_test() -> DeviceInventoryTestGuard {
    static TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    let guard = TEST_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    sk_clear_device_inventory();
    DeviceInventoryTestGuard { _guard: guard }
}
