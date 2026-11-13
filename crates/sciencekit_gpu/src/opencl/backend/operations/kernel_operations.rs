//! Typed kernel operations over host views (spec `opencl-backend`).
//!
//! Each method validates shapes and scalar support, uploads the inputs, runs one
//! kernel, downloads the output and accounts for the bytes moved. The resident
//! path uploads once and reuses a device buffer across calls.

use ndarray::{Array2, ArrayView2};
use sciencekit_common::{SKError, SKFloat};

use super::super::implementation::SKOpenClBackend;
use super::support::{element_bytes, ensure_scalar_supported};
use crate::opencl::buffer::{self, SKOpenClBuffer};
use crate::opencl::kernels::{self, SKOpenClDistanceMetric};
use crate::opencl::observability;

impl SKOpenClBackend {
    /// Upload a host view into a resident buffer (the efficient path).
    pub fn upload_resident<F: SKFloat>(
        &self,
        host: ArrayView2<'_, F>,
    ) -> Result<SKOpenClBuffer<F>, SKError> {
        let span = observability::operation_span("upload_resident");
        let _entered = span.enter();
        let session = self.session()?;
        ensure_scalar_supported::<F>(session)?;
        let resident = buffer::upload_view(session, host)?;
        let bytes = element_bytes::<F>(resident.element_count());
        self.upload_bytes
            .fetch_add(bytes, std::sync::atomic::Ordering::Relaxed);
        observability::record_device(&span, session.device().index);
        observability::record_transfer(&span, bytes);
        Ok(resident)
    }

    /// Download a resident buffer into an owned host matrix.
    pub fn download_resident<F: SKFloat>(
        &self,
        buffer: &SKOpenClBuffer<F>,
    ) -> Result<Array2<F>, SKError> {
        let span = observability::operation_span("download_resident");
        let _entered = span.enter();
        let session = self.session()?;
        let matrix = buffer::download_buffer(session, buffer)?;
        let bytes = element_bytes::<F>(buffer.element_count());
        self.download_bytes
            .fetch_add(bytes, std::sync::atomic::Ordering::Relaxed);
        observability::record_device(&span, session.device().index);
        observability::record_transfer(&span, bytes);
        Ok(matrix)
    }

    /// Pairwise distances between the rows of two matrices.
    pub fn pairwise_distances<F: SKFloat>(
        &self,
        metric: SKOpenClDistanceMetric,
        left: ArrayView2<'_, F>,
        right: ArrayView2<'_, F>,
    ) -> Result<Array2<F>, SKError> {
        let span = observability::operation_span("pairwise_distance");
        let _entered = span.enter();
        let session = self.session()?;
        ensure_scalar_supported::<F>(session)?;
        if left.ncols() != right.ncols() {
            return Err(SKError::ShapeMismatch {
                expected: vec![left.nrows(), left.ncols()],
                found: vec![right.nrows(), right.ncols()],
            });
        }
        let left_resident = buffer::upload_view(session, left)?;
        let right_resident = buffer::upload_view(session, right)?;
        let output = buffer::allocate_resident::<F>(session, (left.nrows(), right.nrows()))?;
        kernels::run_pairwise_distance(session, &left_resident, &right_resident, &output, metric)?;
        let result = buffer::download_buffer(session, &output)?;
        let upload_bytes = element_bytes::<F>(left.len() + right.len());
        let output_bytes = element_bytes::<F>(output.element_count());
        self.charge(upload_bytes, output_bytes);
        observability::record_device(&span, session.device().index);
        observability::record_transfer(&span, upload_bytes + output_bytes);
        Ok(result)
    }

    /// Dense matrix multiplication `left · right`.
    pub fn gemm<F: SKFloat>(
        &self,
        left: ArrayView2<'_, F>,
        right: ArrayView2<'_, F>,
    ) -> Result<Array2<F>, SKError> {
        let span = observability::operation_span("gemm");
        let _entered = span.enter();
        let session = self.session()?;
        ensure_scalar_supported::<F>(session)?;
        if left.ncols() != right.nrows() {
            return Err(SKError::ShapeMismatch {
                expected: vec![left.ncols(), right.ncols()],
                found: vec![right.nrows(), right.ncols()],
            });
        }
        let left_resident = buffer::upload_view(session, left)?;
        let right_resident = buffer::upload_view(session, right)?;
        let output = buffer::allocate_resident::<F>(session, (left.nrows(), right.ncols()))?;
        kernels::run_gemm(session, &left_resident, &right_resident, &output)?;
        let result = buffer::download_buffer(session, &output)?;
        let upload_bytes = element_bytes::<F>(left.len() + right.len());
        let output_bytes = element_bytes::<F>(output.element_count());
        self.charge(upload_bytes, output_bytes);
        observability::record_device(&span, session.device().index);
        observability::record_transfer(&span, upload_bytes + output_bytes);
        Ok(result)
    }
}
