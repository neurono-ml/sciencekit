//! The operations a backend without a device reports (spec `opencl-backend`,
//! requirement *CPU fallback*).
//!
//! With the `opencl` feature off the module compiles to these methods only: no
//! OpenCL symbol is linked or called, and every operation returns the structured
//! unavailability error that names the disabled feature. The typed kernels do
//! not exist in this configuration because no device buffer can hold data, which
//! is why only the contract surface is defined here.

use ndarray::{Array2, ArrayView2};
use sciencekit_common::{SKError, SKFloat, SKGpuKernel};

use super::implementation::SKOpenClBackend;
use crate::buffer::SKDeviceBuffer;
use crate::kernel::SKDeviceBufferRef;

impl SKOpenClBackend {
    /// Upload is unavailable without the feature.
    pub(crate) fn upload_implementation<F: SKFloat>(
        &self,
        _host: ArrayView2<'_, F>,
        _span: &tracing::Span,
    ) -> Result<SKDeviceBuffer<F>, SKError> {
        Err(self.unavailable_error())
    }

    /// Download is unavailable without the feature.
    pub(crate) fn download_implementation<F: SKFloat>(
        &self,
        _buffer: &SKDeviceBuffer<F>,
        _span: &tracing::Span,
    ) -> Result<Array2<F>, SKError> {
        Err(self.unavailable_error())
    }

    /// Dispatch is unavailable without the feature.
    pub(crate) fn dispatch_implementation<F: SKFloat>(
        &self,
        _kernel: SKGpuKernel,
        _arguments: &[SKDeviceBufferRef<'_, F>],
        _span: &tracing::Span,
    ) -> Result<SKDeviceBuffer<F>, SKError> {
        Err(self.unavailable_error())
    }
}
