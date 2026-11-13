//! NumPy `.npy` header parsing (spec `memory-mapped-sources`).
//!
//! The magic, version and header-length framing are validated here; `npyz`
//! parses the Python-literal header itself. The adapter records the payload
//! offset so the mapping can be viewed without copying.

use std::io::Cursor;

use npyz::{DType, Endianness, NpyHeader, Order, TypeChar};
use sciencekit_common::SKError;

const NPY_MAGIC: &[u8; 6] = b"\x93NUMPY";

/// The dtypes the `.npy` adapter reads.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SKNumpyDtype {
    /// `float32` (`<f4`).
    Float32,
    /// `float64` (`<f8`).
    Float64,
    /// `int64` (`<i8`).
    Int64,
}

impl SKNumpyDtype {
    pub(crate) fn name(self) -> &'static str {
        match self {
            SKNumpyDtype::Float32 => "f4",
            SKNumpyDtype::Float64 => "f8",
            SKNumpyDtype::Int64 => "i8",
        }
    }

    pub(crate) fn element_size(self) -> usize {
        match self {
            SKNumpyDtype::Float32 => 4,
            SKNumpyDtype::Float64 | SKNumpyDtype::Int64 => 8,
        }
    }
}

/// A parsed `.npy` header: shape, dtype and the payload offset.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct SKNumpyHeader {
    pub(crate) number_of_rows: usize,
    pub(crate) number_of_columns: usize,
    pub(crate) dtype: SKNumpyDtype,
    pub(crate) data_offset: usize,
}

/// Parse a `.npy` header from the mapped bytes.
pub(crate) fn sk_parse_numpy_header(bytes: &[u8], path: &str) -> Result<SKNumpyHeader, SKError> {
    let data_offset = sk_numpy_data_offset(bytes, path)?;
    let header = NpyHeader::from_reader(Cursor::new(bytes)).map_err(|error| {
        SKError::MalformedMappedHeader {
            path: path.to_string(),
            format: "npy",
            reason: error.to_string(),
        }
    })?;
    if header.order() == Order::Fortran {
        return Err(SKError::UnsupportedOperation {
            operation: "open_numpy_memory_map",
            reason:
                "the array declares Fortran order; save it with `numpy.ascontiguousarray` first"
                    .to_string(),
        });
    }
    let (number_of_rows, number_of_columns) = sk_numpy_shape(header.shape(), path)?;
    let dtype = sk_numpy_dtype(&header.dtype(), path)?;
    Ok(SKNumpyHeader {
        number_of_rows,
        number_of_columns,
        dtype,
        data_offset,
    })
}

fn sk_numpy_data_offset(bytes: &[u8], path: &str) -> Result<usize, SKError> {
    if bytes.len() < 10 || &bytes[..6] != NPY_MAGIC {
        return Err(SKError::MalformedMappedHeader {
            path: path.to_string(),
            format: "npy",
            reason: "the file does not start with the NumPy magic `\\x93NUMPY`".to_string(),
        });
    }
    let major = bytes[6];
    let header_length = if major == 1 {
        u16::from_le_bytes([bytes[8], bytes[9]]) as usize
    } else {
        if bytes.len() < 12 {
            return Err(SKError::MalformedMappedHeader {
                path: path.to_string(),
                format: "npy",
                reason: "the file ends inside the version-2/3 header length".to_string(),
            });
        }
        u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize
    };
    let framing = if major == 1 { 10 } else { 12 };
    let data_offset = framing + header_length;
    if data_offset > bytes.len() {
        return Err(SKError::MalformedMappedHeader {
            path: path.to_string(),
            format: "npy",
            reason: format!(
                "the header declares {header_length} bytes but the file ends at {}",
                bytes.len()
            ),
        });
    }
    Ok(data_offset)
}

fn sk_numpy_shape(shape: &[u64], path: &str) -> Result<(usize, usize), SKError> {
    match shape {
        [rows, columns] => {
            let rows = usize::try_from(*rows).map_err(|_| SKError::MalformedMappedHeader {
                path: path.to_string(),
                format: "npy",
                reason: "the row count does not fit the platform's address space".to_string(),
            })?;
            let columns =
                usize::try_from(*columns).map_err(|_| SKError::MalformedMappedHeader {
                    path: path.to_string(),
                    format: "npy",
                    reason: "the column count does not fit the platform's address space"
                        .to_string(),
                })?;
            Ok((rows, columns))
        }
        other => Err(SKError::MalformedMappedHeader {
            path: path.to_string(),
            format: "npy",
            reason: format!(
                "expected a two-dimensional matrix, found shape {other:?}; one-dimensional arrays are not memory-mapped matrices"
            ),
        }),
    }
}

fn sk_numpy_dtype(dtype: &DType, path: &str) -> Result<SKNumpyDtype, SKError> {
    let DType::Plain(type_string) = dtype else {
        return Err(sk_unsupported_dtype(format!("{dtype:?}"), path));
    };
    if type_string.endianness() == Endianness::Big && type_string.size_field() > 1 {
        return Err(SKError::NonLittleEndianMappedData {
            path: path.to_string(),
            endianness: "big".to_string(),
        });
    }
    match (type_string.type_char(), type_string.size_field()) {
        (TypeChar::Float, 4) => Ok(SKNumpyDtype::Float32),
        (TypeChar::Float, 8) => Ok(SKNumpyDtype::Float64),
        (TypeChar::Int, 8) => Ok(SKNumpyDtype::Int64),
        _ => Err(sk_unsupported_dtype(format!("{dtype:?}"), path)),
    }
}

fn sk_unsupported_dtype(dtype: String, path: &str) -> SKError {
    SKError::UnsupportedMappedDtype {
        path: path.to_string(),
        dtype,
        supported: "float32, float64 and int64 (int64 only through the casting path)".to_string(),
    }
}
