//! The typed device allocation behind one resident buffer (spec
//! `opencl-backend`, requirement *OpenCL buffers*).

use std::ptr;

use num_traits::NumCast;
use opencl3::memory::{Buffer, CL_MEM_READ_WRITE};
use opencl3::types::{CL_BLOCKING, cl_double, cl_float};
use sciencekit_common::{SKError, SKFloat};

use crate::opencl::availability::unavailable_error;
use crate::opencl::session::SKOpenClSession;

/// The typed device storage behind one allocation.
#[derive(Debug)]
enum SKOpenClStorage {
    /// Single-precision storage.
    SinglePrecision(Buffer<cl_float>),
    /// Double-precision storage.
    DoublePrecision(Buffer<cl_double>),
}

/// One device allocation with its element count.
#[derive(Debug)]
pub(crate) struct SKOpenClAllocation {
    length: usize,
    storage: SKOpenClStorage,
}

impl SKOpenClAllocation {
    /// Allocate `element_count` elements of `scalar_type` on the session device.
    pub(crate) fn allocate(
        session: &SKOpenClSession,
        element_count: usize,
        scalar_type: &'static str,
    ) -> Result<Self, SKError> {
        let length = element_count.max(1);
        let bytes_per_element = match scalar_type {
            "f64" => std::mem::size_of::<cl_double>(),
            _ => std::mem::size_of::<cl_float>(),
        };
        let device_memory = session.device().memory_bytes;
        if device_memory > 0
            && (length as u64).saturating_mul(bytes_per_element as u64) > device_memory
        {
            return Err(unavailable_error(format!(
                "the requested allocation of {length} elements exceeds the device's {device_memory} bytes of memory"
            )));
        }
        let storage = match scalar_type {
            "f32" => SKOpenClStorage::SinglePrecision(
                unsafe {
                    Buffer::<cl_float>::create(
                        session.context(),
                        CL_MEM_READ_WRITE,
                        length,
                        ptr::null_mut(),
                    )
                }
                .map_err(allocation_failure)?,
            ),
            "f64" => SKOpenClStorage::DoublePrecision(
                unsafe {
                    Buffer::<cl_double>::create(
                        session.context(),
                        CL_MEM_READ_WRITE,
                        length,
                        ptr::null_mut(),
                    )
                }
                .map_err(allocation_failure)?,
            ),
            other => {
                return Err(SKError::UnsupportedScalarType {
                    backend: "opencl",
                    scalar_type: other,
                });
            }
        };
        Ok(SKOpenClAllocation { length, storage })
    }

    /// The number of elements the allocation can hold.
    pub(crate) fn element_count(&self) -> usize {
        self.length
    }

    /// Copy `values` from the host into this allocation.
    pub(crate) fn write_from_host<F: SKFloat>(
        &mut self,
        session: &SKOpenClSession,
        values: &[F],
    ) -> Result<(), SKError> {
        if values.len() > self.length {
            return Err(SKError::ShapeMismatch {
                expected: vec![self.length],
                found: vec![values.len()],
            });
        }
        match &mut self.storage {
            SKOpenClStorage::SinglePrecision(buffer) => {
                let converted: Vec<cl_float> = values
                    .iter()
                    .map(|value| value.to_f32().unwrap_or(0.0))
                    .collect();
                write_values(session, buffer, &converted)
            }
            SKOpenClStorage::DoublePrecision(buffer) => {
                let converted: Vec<cl_double> = values
                    .iter()
                    .map(|value| value.to_f64().unwrap_or(0.0))
                    .collect();
                write_values(session, buffer, &converted)
            }
        }
    }

    /// Copy the first `element_count` device values back to the host.
    pub(crate) fn read_into_host<F: SKFloat>(
        &self,
        session: &SKOpenClSession,
        element_count: usize,
    ) -> Result<Vec<F>, SKError> {
        match &self.storage {
            SKOpenClStorage::SinglePrecision(buffer) => {
                let values = read_values(session, buffer, element_count)?;
                Ok(values
                    .into_iter()
                    .map(|value| NumCast::from(value).unwrap_or_else(F::zero))
                    .collect())
            }
            SKOpenClStorage::DoublePrecision(buffer) => {
                let values = read_values(session, buffer, element_count)?;
                Ok(values
                    .into_iter()
                    .map(|value| NumCast::from(value).unwrap_or_else(F::zero))
                    .collect())
            }
        }
    }

    /// The single-precision storage, when this is a single-precision allocation.
    pub(crate) fn single_precision(&self) -> Option<&Buffer<cl_float>> {
        match &self.storage {
            SKOpenClStorage::SinglePrecision(buffer) => Some(buffer),
            SKOpenClStorage::DoublePrecision(_) => None,
        }
    }

    /// The double-precision storage, when this is a double-precision allocation.
    pub(crate) fn double_precision(&self) -> Option<&Buffer<cl_double>> {
        match &self.storage {
            SKOpenClStorage::SinglePrecision(_) => None,
            SKOpenClStorage::DoublePrecision(buffer) => Some(buffer),
        }
    }
}

/// The structured error for a failed allocation.
fn allocation_failure(code: impl std::fmt::Debug) -> SKError {
    unavailable_error(format!(
        "OpenCL buffer allocation failed with code {code:?}"
    ))
}

fn write_values<T: Copy>(
    session: &SKOpenClSession,
    buffer: &mut Buffer<T>,
    values: &[T],
) -> Result<(), SKError> {
    unsafe {
        session
            .queue()
            .enqueue_write_buffer(buffer, CL_BLOCKING, 0, values, &[])
    }
    .map_err(|code| SKError::DeviceTransferFailure {
        reason: format!("OpenCL buffer write failed with code {code}"),
    })?;
    Ok(())
}

fn read_values<T: Copy + Default>(
    session: &SKOpenClSession,
    buffer: &Buffer<T>,
    element_count: usize,
) -> Result<Vec<T>, SKError> {
    let mut values: Vec<T> = (0..element_count).map(|_| T::default()).collect();
    unsafe {
        session
            .queue()
            .enqueue_read_buffer(buffer, CL_BLOCKING, 0, &mut values, &[])
    }
    .map_err(|code| SKError::DeviceTransferFailure {
        reason: format!("OpenCL buffer read failed with code {code}"),
    })?;
    Ok(values)
}
