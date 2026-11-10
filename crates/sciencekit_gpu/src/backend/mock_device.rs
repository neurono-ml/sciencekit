//! A mock device backend driving the contract tests.
//!
//! The mock stands in for a real runtime: it owns a fixed inventory, can be
//! told to fail discovery, and executes buffers and kernels in host memory. Its
//! primary device — the one uploads target — is the first enumerated device. It
//! is compiled only for tests, so the production surface stays free of mock
//! machinery.

use ndarray::{Array2, ArrayView2};
use sciencekit_common::{
    SKBackendKind, SKDeviceCapabilities, SKDeviceInfo, SKError, SKFloat, SKGpuKernel,
};

use super::contract::SKComputeBackend;
use super::discovery::SKDeviceDiscovery;
use crate::buffer::{SKDeviceBuffer, SKDeviceBufferBuilder};
use crate::kernel::SKDeviceBufferRef;

/// A backend over a caller-supplied inventory.
pub(crate) struct SKMockComputeBackend {
    devices: Vec<SKDeviceInfo>,
    discovery_failure: Option<&'static str>,
    transfer_failure: Option<&'static str>,
    capacity_elements: Option<usize>,
}

impl SKMockComputeBackend {
    /// A backend enumerating exactly `devices`.
    pub(crate) fn new(devices: Vec<SKDeviceInfo>) -> Self {
        SKMockComputeBackend {
            devices,
            discovery_failure: None,
            transfer_failure: None,
            capacity_elements: None,
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
            transfer_failure: None,
            capacity_elements: None,
        }
    }

    /// A backend whose uploads and downloads fail with `reason`.
    pub(crate) fn failing_transfers(mut self, reason: &'static str) -> Self {
        self.transfer_failure = Some(reason);
        self
    }

    /// A backend whose device accepts at most `elements` elements per buffer.
    pub(crate) fn limited_to_elements(mut self, elements: usize) -> Self {
        self.capacity_elements = Some(elements);
        self
    }

    /// The injected transfer failure, if any.
    fn transfer_failure(&self) -> Option<SKError> {
        self.transfer_failure
            .map(|reason| SKError::DeviceTransferFailure {
                reason: reason.to_string(),
            })
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

    /// The device an upload targets: the first one enumerated.
    fn primary_device(&self) -> Result<&SKDeviceInfo, SKError> {
        self.devices
            .first()
            .ok_or_else(|| SKError::DeviceUnavailable {
                backend: "mock",
                reason: "no device is enumerated; the host fallback runs instead".to_string(),
            })
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

impl<F: SKFloat> SKComputeBackend<F> for SKMockComputeBackend {
    fn backend_kind(&self) -> SKBackendKind {
        self.devices
            .first()
            .map(|device| device.kind)
            .unwrap_or(SKBackendKind::Faer)
    }

    fn upload(&self, host: ArrayView2<'_, F>) -> Result<SKDeviceBuffer<F>, SKError> {
        if let Some(failure) = self.transfer_failure() {
            return Err(failure);
        }
        let device = self.primary_device()?;
        let scalar_type = std::any::type_name::<F>();
        if !device.supports_scalar_type(scalar_type) {
            return Err(SKError::UnsupportedScalarType {
                backend: "mock",
                scalar_type,
            });
        }
        let elements = host.nrows().saturating_mul(host.ncols());
        if let Some(capacity) = self.capacity_elements {
            if elements > capacity {
                return Err(SKError::ShapeMismatch {
                    expected: vec![capacity],
                    found: vec![elements],
                });
            }
        }
        SKDeviceBufferBuilder::new(device.kind, device.index)
            .shape(host.nrows(), host.ncols())
            .values(host.iter().copied().collect())
            .build()
    }

    fn download(&self, buffer: &SKDeviceBuffer<F>) -> Result<Array2<F>, SKError> {
        if let Some(failure) = self.transfer_failure() {
            return Err(failure);
        }
        let own_kind = self
            .devices
            .first()
            .map(|device| device.kind)
            .unwrap_or(SKBackendKind::Faer);
        if buffer.backend_kind() != own_kind {
            return Err(SKError::DeviceUnavailable {
                backend: buffer.backend_kind().name(),
                reason: "the buffer belongs to a different backend".to_string(),
            });
        }
        buffer.to_host_matrix()
    }

    fn dispatch(
        &self,
        kernel: SKGpuKernel,
        arguments: &[SKDeviceBufferRef<'_, F>],
    ) -> Result<SKDeviceBuffer<F>, SKError> {
        let device = self.primary_device()?;
        if !device.capabilities.supports_kernel(kernel) {
            return Err(SKError::UnsupportedOperation {
                operation: kernel.name(),
                reason: format!(
                    "the mock device `{}` does not implement this kernel",
                    device.name
                ),
            });
        }
        let input = arguments.first().ok_or_else(|| {
            SKError::Conversion("kernel dispatch requires at least one argument buffer".to_string())
        })?;
        // The mock passes its input through: the contract under test is the
        // dispatch surface, not a numerical kernel.
        Ok((*input).clone())
    }
}
