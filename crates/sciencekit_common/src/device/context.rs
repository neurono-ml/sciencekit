//! The device request an execution context carries (spec
//! `gpu-compute-contracts`).
//!
//! The context is explicit and injectable: planning never enumerates devices
//! itself, it receives the inventory the backend discovered. That keeps
//! resolution pure and cheap to test.

use crate::observability::SKBackendKind;

use super::info::SKDeviceInfo;
use super::kernel::SKGpuKernel;

/// The device request attached to an execution context.
///
/// `configured_backend` is the backend the caller configured for the operation;
/// only a device kind can select a device. `required_kernel` is the kernel the
/// operation needs, when it has one. `device_inventory` is the enumeration the
/// backend produced — empty when no device is present.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SKDeviceContext {
    /// The backend configured for the operation.
    pub configured_backend: SKBackendKind,
    /// The kernel the operation requires, when it requires one.
    pub required_kernel: Option<SKGpuKernel>,
    /// The devices the backend enumerated, in its stable order.
    pub device_inventory: Vec<SKDeviceInfo>,
}

impl SKDeviceContext {
    /// A request for `configured_backend` with an empty inventory.
    pub fn new(configured_backend: SKBackendKind) -> Self {
        SKDeviceContext {
            configured_backend,
            ..SKDeviceContext::default()
        }
    }

    /// Declare the kernel the operation requires.
    pub fn with_required_kernel(mut self, kernel: SKGpuKernel) -> Self {
        self.required_kernel = Some(kernel);
        self
    }

    /// Attach the enumerated device inventory.
    pub fn with_inventory(mut self, inventory: Vec<SKDeviceInfo>) -> Self {
        self.device_inventory = inventory;
        self
    }
}
