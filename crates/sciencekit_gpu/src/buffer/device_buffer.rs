//! A handle to device-resident storage (spec `gpu-compute-contracts`).
//!
//! A buffer carries everything an operation must validate before touching
//! device memory: the backend and device that own it, the scalar type it
//! stores, its logical shape and its capacity in elements. The foundation
//! backs it with an owned staging allocation; a concrete backend replaces the
//! staging vector with its runtime handle behind the same validated surface.

use ndarray::Array2;
use sciencekit_common::{SKBackendKind, SKFloat};

/// Device-resident storage for one dense matrix.
#[derive(Debug, Clone, PartialEq)]
pub struct SKDeviceBuffer<F: SKFloat> {
    backend_kind: SKBackendKind,
    device_index: usize,
    scalar_type: &'static str,
    shape: (usize, usize),
    capacity: usize,
    values: Vec<F>,
}

impl<F: SKFloat> SKDeviceBuffer<F> {
    /// Validate and build a buffer (the only constructor; see
    /// [`SKDeviceBufferBuilder`](super::SKDeviceBufferBuilder)).
    pub(crate) fn from_parts(
        backend_kind: SKBackendKind,
        device_index: usize,
        shape: (usize, usize),
        capacity: usize,
        values: Vec<F>,
    ) -> Result<Self, sciencekit_common::SKError> {
        let elements = shape.0.saturating_mul(shape.1);
        if elements > capacity {
            return Err(sciencekit_common::SKError::ShapeMismatch {
                expected: vec![capacity],
                found: vec![elements],
            });
        }
        if values.len() != elements {
            return Err(sciencekit_common::SKError::ShapeMismatch {
                expected: vec![shape.0, shape.1],
                found: vec![values.len()],
            });
        }
        Ok(SKDeviceBuffer {
            backend_kind,
            device_index,
            scalar_type: F::SCALAR_NAME,
            shape,
            capacity,
            values,
        })
    }

    /// The backend that owns the buffer.
    pub fn backend_kind(&self) -> SKBackendKind {
        self.backend_kind
    }

    /// The device index the buffer was allocated on.
    pub fn device_index(&self) -> usize {
        self.device_index
    }

    /// The Rust name of the stored scalar type (`f32` or `f64`).
    pub fn scalar_type(&self) -> &'static str {
        self.scalar_type
    }

    /// The logical `(rows, columns)` shape.
    pub fn shape(&self) -> (usize, usize) {
        self.shape
    }

    /// The allocated capacity, in elements.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Build an owned host matrix from the buffer's staged values.
    ///
    /// This is the foundation's transfer surface: a concrete backend fills the
    /// staging allocation by copying device memory back, or exposes the mapped
    /// path separately when its capabilities allow it.
    pub fn to_host_matrix(&self) -> Result<Array2<F>, sciencekit_common::SKError> {
        Array2::from_shape_vec(self.shape, self.values.clone())
            .map_err(|error| sciencekit_common::SKError::Conversion(error.to_string()))
    }
}
