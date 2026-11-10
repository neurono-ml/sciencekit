//! The compute-backend contract (spec `gpu-compute-contracts`).
//!
//! A device backend extends [`SKDeviceDiscovery`] with the two explicit
//! transfers of the safe path: [`SKComputeBackend::upload`] copies a host view
//! into device storage and [`SKComputeBackend::download`] copies it back into
//! an owned host matrix. Kernel dispatch is layered on the same contract in
//! `kernel::dispatch`.

use ndarray::{Array2, ArrayView2};
use sciencekit_common::{SKBackendKind, SKFloat};

use super::discovery::SKDeviceDiscovery;
use crate::buffer::SKDeviceBuffer;

/// A backend that drives one or more devices.
///
/// The trait is `Send + Sync` so device discovery and planning are safe under
/// concurrent calls. `Self::Error` converts from [`sciencekit_common::SKError`],
/// keeping the structured taxonomy shared even when a backend adds its own
/// failure variants.
pub trait SKComputeBackend<F: SKFloat>: SKDeviceDiscovery {
    /// The backend kind this implementation drives.
    fn backend_kind(&self) -> SKBackendKind;

    /// Copy a host matrix into device storage.
    fn upload(&self, host: ArrayView2<'_, F>) -> Result<SKDeviceBuffer<F>, Self::Error>;

    /// Copy device storage back into an owned host matrix.
    fn download(&self, buffer: &SKDeviceBuffer<F>) -> Result<Array2<F>, Self::Error>;
}
