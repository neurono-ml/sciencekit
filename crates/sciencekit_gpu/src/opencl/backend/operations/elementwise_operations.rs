//! Elementwise operations over host views (spec `opencl-backend`).

use ndarray::{Array2, ArrayView2};
use sciencekit_common::{SKError, SKFloat};

use super::super::implementation::SKOpenClBackend;
use super::support::{element_bytes, ensure_scalar_supported};
use crate::opencl::buffer;
use crate::opencl::kernels;
use crate::opencl::observability;

impl SKOpenClBackend {
    /// Elementwise affine transformation `input · factor + offset`.
    pub fn transform<F: SKFloat>(
        &self,
        input: ArrayView2<'_, F>,
        factor: F,
        offset: F,
    ) -> Result<Array2<F>, SKError> {
        let span = self.operation_span("elementwise_transform");
        let _entered = span.enter();
        let session = self.session()?;
        ensure_scalar_supported::<F>(session)?;
        let input_resident = buffer::upload_view(session, input)?;
        let output = buffer::allocate_resident::<F>(session, input_resident.shape())?;
        kernels::run_transform(session, &input_resident, &output, factor, offset)?;
        let result = buffer::download_buffer(session, &output)?;
        let upload_bytes = element_bytes::<F>(input.len());
        let output_bytes = element_bytes::<F>(output.element_count());
        self.charge(upload_bytes, output_bytes);
        observability::record_device(&span, session.device().index);
        observability::record_transfer(&span, upload_bytes + output_bytes);
        Ok(result)
    }

    /// Elementwise combination `left + right`.
    pub fn combine<F: SKFloat>(
        &self,
        left: ArrayView2<'_, F>,
        right: ArrayView2<'_, F>,
    ) -> Result<Array2<F>, SKError> {
        let span = self.operation_span("elementwise_combine");
        let _entered = span.enter();
        let session = self.session()?;
        ensure_scalar_supported::<F>(session)?;
        if left.shape() != right.shape() {
            return Err(SKError::ShapeMismatch {
                expected: vec![left.nrows(), left.ncols()],
                found: vec![right.nrows(), right.ncols()],
            });
        }
        let left_resident = buffer::upload_view(session, left)?;
        let right_resident = buffer::upload_view(session, right)?;
        let output = buffer::allocate_resident::<F>(session, left_resident.shape())?;
        kernels::run_combine(session, &left_resident, &right_resident, &output)?;
        let result = buffer::download_buffer(session, &output)?;
        let upload_bytes = element_bytes::<F>(left.len() + right.len());
        let output_bytes = element_bytes::<F>(output.element_count());
        self.charge(upload_bytes, output_bytes);
        observability::record_device(&span, session.device().index);
        observability::record_transfer(&span, upload_bytes + output_bytes);
        Ok(result)
    }
}
