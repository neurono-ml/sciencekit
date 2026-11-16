//! The contract implementations of the OpenCL backend (spec `opencl-backend`).
//!
//! `SKDeviceDiscovery` enumerates through the ICD loader; `SKComputeBackend`
//! routes upload, download and dispatch to the device operations. Every entry
//! goes through [`SKOpenClBackend::run_operation`], which opens exactly one
//! observability span before any availability check, so the trace records
//! `opencl` when a device serves the call and `faer` plus the fallback reason
//! when it does not.

use ndarray::{Array2, ArrayView2};
use sciencekit_common::{SKBackendKind, SKDeviceInfo, SKError, SKFloat, SKGpuKernel};

use super::implementation::SKOpenClBackend;
use crate::backend::{SKComputeBackend, SKDeviceDiscovery};
use crate::buffer::{SKDeviceBuffer, SKDeviceTransferPolicy};
use crate::kernel::SKDeviceBufferRef;
use crate::opencl::device::sk_opencl_devices;

#[cfg(feature = "opencl")]
use crate::buffer::SKDeviceBufferBuilder;

impl SKDeviceDiscovery for SKOpenClBackend {
    type Error = SKError;

    fn discover_devices(&self) -> Result<Vec<SKDeviceInfo>, SKError> {
        self.run_operation("discover_devices", |_span| sk_opencl_devices())
    }
}

impl<F: SKFloat> SKComputeBackend<F> for SKOpenClBackend {
    fn backend_kind(&self) -> SKBackendKind {
        SKBackendKind::OpenCl
    }

    fn transfer_policy(&self) -> SKDeviceTransferPolicy {
        self.transfer_policy
    }

    fn upload(&self, host: ArrayView2<'_, F>) -> Result<SKDeviceBuffer<F>, SKError> {
        self.run_operation("upload", |span| self.upload_implementation(host, span))
    }

    fn download(&self, buffer: &SKDeviceBuffer<F>) -> Result<Array2<F>, SKError> {
        self.run_operation("download", |span| {
            self.download_implementation(buffer, span)
        })
    }

    fn dispatch(
        &self,
        kernel: SKGpuKernel,
        arguments: &[SKDeviceBufferRef<'_, F>],
    ) -> Result<SKDeviceBuffer<F>, SKError> {
        self.run_operation("dispatch", |span| {
            self.dispatch_implementation(kernel, arguments, span)
        })
    }
}

/// Build a staged contract buffer from a host matrix.
#[cfg(feature = "opencl")]
pub(crate) fn staged_buffer<F: SKFloat>(
    matrix: &Array2<F>,
    device_index: usize,
) -> Result<SKDeviceBuffer<F>, SKError> {
    SKDeviceBufferBuilder::new(SKBackendKind::OpenCl, device_index)
        .shape(matrix.nrows(), matrix.ncols())
        .values(matrix.iter().copied().collect())
        .build()
}
