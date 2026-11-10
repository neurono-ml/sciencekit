//! Borrowed ABI descriptors built over the copied numeric buffers.

use std::ffi::c_void;
use std::ptr;

use sciencekit_ffi::{SKFfiArray, SKFfiDataType};

use crate::array::{SKLuaMatrix, SKLuaVector};

/// The shape and values behind one borrowed [`SKFfiArray`].
///
/// The descriptor must outlive the ABI call it is passed to; the borrow is
/// enforced by keeping both in the same scope at the call site.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct SKLuaDescriptor {
    shape: Vec<i64>,
    values: Vec<f64>,
}

impl SKLuaDescriptor {
    /// Describe a matrix as a two-dimensional descriptor.
    pub(crate) fn matrix(matrix: &SKLuaMatrix) -> SKLuaDescriptor {
        SKLuaDescriptor {
            shape: vec![matrix.rows() as i64, matrix.columns() as i64],
            values: matrix.values().to_vec(),
        }
    }

    /// Describe a vector as a one-dimensional descriptor.
    pub(crate) fn vector(vector: &SKLuaVector) -> SKLuaDescriptor {
        SKLuaDescriptor {
            shape: vec![vector.len() as i64],
            values: vector.values().to_vec(),
        }
    }

    /// Borrow the descriptor as the C-layout struct the ABI reads.
    pub(crate) fn as_ffi_array(&self) -> SKFfiArray {
        SKFfiArray {
            dtype: SKFfiDataType::F64 as i32,
            ndim: self.shape.len() as i32,
            shape: self.shape.as_ptr(),
            strides: ptr::null(),
            data: self.values.as_ptr() as *const c_void,
            len: self.values.len() as i64,
        }
    }
}
