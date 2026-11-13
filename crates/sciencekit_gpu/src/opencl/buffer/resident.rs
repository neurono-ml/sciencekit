//! Validated device-resident matrices and their transfers (spec
//! `opencl-backend`, requirement *OpenCL buffers*).
//!
//! A resident buffer owns a real device allocation plus the metadata an
//! operation must validate: shape, scalar type and owning device. Uploads copy
//! the host view into device memory; downloads copy it back. The safe path
//! always copies — the mapped path is advertised through the capability flags
//! but the backend keeps the explicit, observable copy as its default.

use std::marker::PhantomData;
use std::sync::Arc;

use ndarray::{Array2, ArrayView2};
use sciencekit_common::{SKError, SKFloat};

use super::allocation::SKOpenClAllocation;
use crate::opencl::session::SKOpenClSession;

/// A validated, device-resident matrix of the caller's scalar type.
pub struct SKOpenClBuffer<F: SKFloat> {
    allocation: Arc<SKOpenClAllocation>,
    shape: (usize, usize),
    device_index: usize,
    scalar_type: &'static str,
    marker: PhantomData<F>,
}

impl<F: SKFloat> SKOpenClBuffer<F> {
    /// Build a resident buffer around an allocation.
    pub(crate) fn from_allocation(
        allocation: SKOpenClAllocation,
        shape: (usize, usize),
        device_index: usize,
    ) -> Self {
        SKOpenClBuffer {
            allocation: Arc::new(allocation),
            shape,
            device_index,
            scalar_type: F::SCALAR_NAME,
            marker: PhantomData,
        }
    }

    /// The logical `(rows, columns)` shape.
    pub fn shape(&self) -> (usize, usize) {
        self.shape
    }

    /// The scalar type the buffer stores, by its Rust name.
    pub fn scalar_type(&self) -> &'static str {
        self.scalar_type
    }

    /// The device index that owns the buffer.
    pub fn device_index(&self) -> usize {
        self.device_index
    }

    /// The number of elements the allocation holds.
    pub fn element_count(&self) -> usize {
        self.allocation.element_count()
    }

    /// The element count implied by the logical shape, rejecting overflow.
    pub(crate) fn element_count_of_shape(&self) -> Result<usize, SKError> {
        self.shape
            .0
            .checked_mul(self.shape.1)
            .ok_or_else(|| SKError::ShapeMismatch {
                expected: vec![usize::MAX],
                found: vec![self.shape.0, self.shape.1],
            })
    }

    /// The underlying device allocation.
    pub(crate) fn allocation(&self) -> &SKOpenClAllocation {
        &self.allocation
    }
}

/// Upload a host view, returning a device-resident buffer.
pub(crate) fn upload_view<F: SKFloat>(
    session: &SKOpenClSession,
    host: ArrayView2<'_, F>,
) -> Result<SKOpenClBuffer<F>, SKError> {
    let shape = (host.nrows(), host.ncols());
    let values: Vec<F> = host.iter().copied().collect();
    let mut allocation = SKOpenClAllocation::allocate(session, values.len(), F::SCALAR_NAME)?;
    allocation.write_from_host(session, &values)?;
    Ok(SKOpenClBuffer::from_allocation(
        allocation,
        shape,
        session.device().index,
    ))
}

/// Allocate a resident buffer without uploading values (a kernel output).
pub(crate) fn allocate_resident<F: SKFloat>(
    session: &SKOpenClSession,
    shape: (usize, usize),
) -> Result<SKOpenClBuffer<F>, SKError> {
    let element_count = shape
        .0
        .checked_mul(shape.1)
        .ok_or_else(|| SKError::ShapeMismatch {
            expected: vec![usize::MAX],
            found: vec![shape.0, shape.1],
        })?;
    let allocation = SKOpenClAllocation::allocate(session, element_count, F::SCALAR_NAME)?;
    Ok(SKOpenClBuffer::from_allocation(
        allocation,
        shape,
        session.device().index,
    ))
}

/// Download a resident buffer into an owned host matrix.
pub(crate) fn download_buffer<F: SKFloat>(
    session: &SKOpenClSession,
    buffer: &SKOpenClBuffer<F>,
) -> Result<Array2<F>, SKError> {
    let element_count = buffer.shape().0 * buffer.shape().1;
    let values = buffer
        .allocation()
        .read_into_host::<F>(session, element_count)?;
    Array2::from_shape_vec(buffer.shape(), values)
        .map_err(|error| SKError::Conversion(error.to_string()))
}

/// Upload an owned host matrix (used by the contract surface).
pub(crate) fn upload_owned<F: SKFloat>(
    session: &SKOpenClSession,
    host: &Array2<F>,
) -> Result<SKOpenClBuffer<F>, SKError> {
    upload_view(session, host.view())
}
