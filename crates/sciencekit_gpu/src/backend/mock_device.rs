//! A mock device backend driving the contract tests.
//!
//! The mock stands in for a real runtime: it owns a fixed inventory, can be
//! told to fail discovery, and (in later tasks) executes buffers and kernels in
//! host memory. It is compiled only for tests, so the production surface stays
//! free of mock machinery.

use sciencekit_common::{SKBackendKind, SKDeviceCapabilities, SKDeviceInfo, SKError};

use super::discovery::SKDeviceDiscovery;

/// A backend over a caller-supplied inventory.
pub(crate) struct SKMockComputeBackend {
    devices: Vec<SKDeviceInfo>,
    discovery_failure: Option<&'static str>,
}

impl SKMockComputeBackend {
    /// A backend enumerating exactly `devices`.
    pub(crate) fn new(devices: Vec<SKDeviceInfo>) -> Self {
        SKMockComputeBackend {
            devices,
            discovery_failure: None,
        }
    }

    /// A backend with no device at all.
    pub(crate) fn without_devices() -> Self {
        SKMockComputeBackend::new(Vec::new())
    }

    /// A backend whose discovery always fails with `reason`.
    pub(crate) fn failing_discovery(reason: &'static str) -> Self {
        SKMockComputeBackend {
            devices: Vec::new(),
            discovery_failure: Some(reason),
        }
    }

    /// Build one device description for an inventory.
    pub(crate) fn device(
        index: usize,
        kind: SKBackendKind,
        name: &'static str,
        memory_bytes: u64,
        capabilities: SKDeviceCapabilities,
    ) -> SKDeviceInfo {
        SKDeviceInfo::new(index, kind, name, memory_bytes, capabilities)
    }
}

impl SKDeviceDiscovery for SKMockComputeBackend {
    type Error = SKError;

    fn discover_devices(&self) -> Result<Vec<SKDeviceInfo>, SKError> {
        match self.discovery_failure {
            Some(reason) => Err(SKError::DeviceUnavailable {
                backend: "mock",
                reason: reason.to_string(),
            }),
            None => Ok(self.devices.clone()),
        }
    }
}
