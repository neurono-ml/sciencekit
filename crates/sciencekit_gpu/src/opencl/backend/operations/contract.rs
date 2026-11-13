//! Contract-level operations: upload, download and kernel dispatch (spec
//! `opencl-backend`).
//!
//! The shared contract carries staged buffers without a runtime handle, so the
//! contract path performs a genuine device round-trip for every call: the host
//! view is written to device memory, read back and staged. Callers that hold a
//! device handle use the resident path in `kernel_operations` instead and avoid
//! the extra copies.

use ndarray::{Array2, ArrayView2};
use sciencekit_common::{SKBackendKind, SKError, SKFloat, SKGpuKernel};

use super::super::contract_impl::staged_buffer;
use super::super::implementation::SKOpenClBackend;
use super::support::{argument_matrix, element_bytes, ensure_scalar_supported};
use crate::buffer::SKDeviceBuffer;
use crate::kernel::SKDeviceBufferRef;
use crate::opencl::buffer;
use crate::opencl::kernels::SKOpenClDistanceMetric;
use crate::opencl::observability;

impl SKOpenClBackend {
    /// Upload through contract buffers: device write, device read, staged copy.
    pub(crate) fn upload_implementation<F: SKFloat>(
        &self,
        host: ArrayView2<'_, F>,
        span: &tracing::Span,
    ) -> Result<SKDeviceBuffer<F>, SKError> {
        let session = self.session()?;
        ensure_scalar_supported::<F>(session)?;
        let resident = buffer::upload_view(session, host)?;
        let round_trip = buffer::download_buffer(session, &resident)?;
        let bytes = element_bytes::<F>(resident.element_count());
        self.upload_bytes
            .fetch_add(bytes, std::sync::atomic::Ordering::Relaxed);
        self.download_bytes
            .fetch_add(bytes, std::sync::atomic::Ordering::Relaxed);
        observability::record_device(span, session.device().index);
        observability::record_transfer(span, bytes.saturating_mul(2));
        staged_buffer(&round_trip, session.device().index)
    }

    /// Download through contract buffers: device upload, device read, host copy.
    pub(crate) fn download_implementation<F: SKFloat>(
        &self,
        buffer: &SKDeviceBuffer<F>,
        span: &tracing::Span,
    ) -> Result<Array2<F>, SKError> {
        let session = self.session()?;
        ensure_scalar_supported::<F>(session)?;
        if buffer.backend_kind() != SKBackendKind::OpenCl
            || buffer.device_index() != session.device().index
        {
            return Err(SKError::DeviceTransferFailure {
                reason: "the buffer belongs to a different device".to_string(),
            });
        }
        let host = buffer.to_host_matrix()?;
        let resident = buffer::upload_owned(session, &host)?;
        let result = buffer::download_buffer(session, &resident)?;
        let bytes = element_bytes::<F>(resident.element_count());
        self.upload_bytes
            .fetch_add(bytes, std::sync::atomic::Ordering::Relaxed);
        self.download_bytes
            .fetch_add(bytes, std::sync::atomic::Ordering::Relaxed);
        observability::record_device(span, session.device().index);
        observability::record_transfer(span, bytes.saturating_mul(2));
        Ok(result)
    }

    /// Dispatch a catalogue kernel over contract buffers.
    pub(crate) fn dispatch_implementation<F: SKFloat>(
        &self,
        kernel: SKGpuKernel,
        arguments: &[SKDeviceBufferRef<'_, F>],
        span: &tracing::Span,
    ) -> Result<SKDeviceBuffer<F>, SKError> {
        let session = self.session()?;
        ensure_scalar_supported::<F>(session)?;
        let operation = match kernel {
            SKGpuKernel::PairwiseDistance => {
                let left = argument_matrix(arguments, 0, "pairwise-distance")?;
                let right = argument_matrix(arguments, 1, "pairwise-distance")?;
                self.pairwise_distances(
                    SKOpenClDistanceMetric::Euclidean,
                    left.view(),
                    right.view(),
                )?
            }
            SKGpuKernel::Gemm => {
                let left = argument_matrix(arguments, 0, "gemm")?;
                let right = argument_matrix(arguments, 1, "gemm")?;
                self.gemm(left.view(), right.view())?
            }
            SKGpuKernel::Elementwise => {
                let left = argument_matrix(arguments, 0, "elementwise")?;
                let right = argument_matrix(arguments, 1, "elementwise")?;
                self.combine(left.view(), right.view())?
            }
            SKGpuKernel::TreePredict => {
                let features = argument_matrix(arguments, 0, "tree-predict")?;
                let nodes = argument_matrix(arguments, 1, "tree-predict")?;
                self.predict_trees(features.view(), nodes.view())?
            }
        };
        observability::record_device(span, session.device().index);
        observability::record_transfer(span, element_bytes::<F>(operation.len()));
        self.kernel_bytes.fetch_add(
            element_bytes::<F>(operation.len()),
            std::sync::atomic::Ordering::Relaxed,
        );
        staged_buffer(&operation, session.device().index)
    }
}
