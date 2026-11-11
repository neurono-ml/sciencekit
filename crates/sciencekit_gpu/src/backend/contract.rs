//! The compute-backend contract (spec `gpu-compute-contracts`).
//!
//! A device backend extends [`SKDeviceDiscovery`] with the two explicit
//! transfers of the safe path: [`SKComputeBackend::upload`] copies a host view
//! into device storage and [`SKComputeBackend::download`] copies it back into
//! an owned host matrix. Kernel dispatch is layered on the same contract in
//! `kernel::dispatch`.

use ndarray::{Array2, ArrayView2};
use sciencekit_common::{SKBackendKind, SKFloat, SKGpuKernel};

use super::discovery::SKDeviceDiscovery;
use crate::buffer::{SKDeviceBuffer, SKDeviceTransferPolicy};
use crate::kernel::SKDeviceBufferRef;

/// A backend that drives one or more devices.
///
/// The trait is `Send + Sync` so device discovery and planning are safe under
/// concurrent calls. `Self::Error` converts from [`sciencekit_common::SKError`],
/// keeping the structured taxonomy shared even when a backend adds its own
/// failure variants.
pub trait SKComputeBackend<F: SKFloat>: SKDeviceDiscovery {
    /// The backend kind this implementation drives.
    fn backend_kind(&self) -> SKBackendKind;

    /// How this backend moves buffers between host and device memory.
    ///
    /// Defaults to [`SKDeviceTransferPolicy::Explicit`]: transfers are explicit
    /// host↔device copies unless a backend opts into mapped memory.
    fn transfer_policy(&self) -> SKDeviceTransferPolicy {
        SKDeviceTransferPolicy::Explicit
    }

    /// Copy a host matrix into device storage.
    fn upload(&self, host: ArrayView2<'_, F>) -> Result<SKDeviceBuffer<F>, Self::Error>;

    /// Copy device storage back into an owned host matrix.
    fn download(&self, buffer: &SKDeviceBuffer<F>) -> Result<Array2<F>, Self::Error>;

    /// Run `kernel` over `arguments` on the device and return the result buffer.
    ///
    /// A backend that does not implement the kernel returns
    /// [`SKError::UnsupportedOperation`](sciencekit_common::SKError::UnsupportedOperation)
    /// naming it; it never silently computes a different kernel.
    fn dispatch(
        &self,
        kernel: SKGpuKernel,
        arguments: &[SKDeviceBufferRef<'_, F>],
    ) -> Result<SKDeviceBuffer<F>, Self::Error>;
}
