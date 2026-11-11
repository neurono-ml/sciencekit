//! The description of one enumerated device.

use crate::observability::SKBackendKind;

use super::capabilities::SKDeviceCapabilities;

/// The deterministic description of one device in an inventory.
///
/// `index` is the enumeration position within the device's backend kind — the
/// stable, backend-assigned identity the planner sorts by when several devices
/// of the same kind are present. `kind` names the compute backend that drives
/// the device, so a device is always reachable through exactly one backend.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SKDeviceInfo {
    /// The enumeration index within the device's backend kind.
    pub index: usize,
    /// The backend kind that drives this device.
    pub kind: SKBackendKind,
    /// The human-readable device name reported by the runtime.
    pub name: String,
    /// The total device memory, in bytes.
    pub memory_bytes: u64,
    /// The capabilities the device advertises.
    pub capabilities: SKDeviceCapabilities,
}

impl SKDeviceInfo {
    /// Describe one device.
    pub fn new(
        index: usize,
        kind: SKBackendKind,
        name: impl Into<String>,
        memory_bytes: u64,
        capabilities: SKDeviceCapabilities,
    ) -> Self {
        SKDeviceInfo {
            index,
            kind,
            name: name.into(),
            memory_bytes,
            capabilities,
        }
    }

    /// Whether this device is driven by a device backend kind.
    pub fn is_device(&self) -> bool {
        self.kind.is_device()
    }

    /// Whether the device supports the scalar type named by its Rust name.
    pub fn supports_scalar_type(&self, scalar_type: &str) -> bool {
        self.capabilities.supports_scalar_type(scalar_type)
    }
}
