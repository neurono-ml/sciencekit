//! Builder for validated device buffers (spec `gpu-compute-contracts`).

use sciencekit_common::{SKBackendKind, SKError, SKFloat};

use super::device_buffer::SKDeviceBuffer;

/// Builds a [`SKDeviceBuffer`], validating shape, capacity and payload length.
///
/// The builder is the only public constructor: a buffer can never exist with a
/// shape that exceeds its capacity or with a payload that does not match its
/// shape.
#[derive(Debug, Clone)]
pub struct SKDeviceBufferBuilder<F: SKFloat> {
    backend_kind: SKBackendKind,
    device_index: usize,
    shape: (usize, usize),
    capacity: Option<usize>,
    values: Vec<F>,
}

impl<F: SKFloat> SKDeviceBufferBuilder<F> {
    /// Start a buffer for the given backend and device.
    pub fn new(backend_kind: SKBackendKind, device_index: usize) -> Self {
        SKDeviceBufferBuilder {
            backend_kind,
            device_index,
            shape: (0, 0),
            capacity: None,
            values: Vec::new(),
        }
    }

    /// Set the logical `(rows, columns)` shape.
    pub fn shape(mut self, rows: usize, columns: usize) -> Self {
        self.shape = (rows, columns);
        self
    }

    /// Set the allocated capacity in elements; defaults to the shape size.
    pub fn capacity(mut self, elements: usize) -> Self {
        self.capacity = Some(elements);
        self
    }

    /// Set the row-major payload.
    pub fn values(mut self, values: Vec<F>) -> Self {
        self.values = values;
        self
    }

    /// Validate and build the buffer.
    pub fn build(self) -> Result<SKDeviceBuffer<F>, SKError> {
        let elements = self.shape.0.saturating_mul(self.shape.1);
        let capacity = self.capacity.unwrap_or(elements);
        SKDeviceBuffer::from_parts(
            self.backend_kind,
            self.device_index,
            self.shape,
            capacity,
            self.values,
        )
    }
}
