//! The descriptor validation error and its message.

use std::fmt;

/// A malformed descriptor or an incompatible argument.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SKFfiArrayError {
    /// The data pointer is null.
    NullData,
    /// The shape pointer is null.
    NullShape,
    /// A descriptor pointer is not aligned for its target type: `data` for the
    /// declared element type, `shape` and `strides` for `int64_t`.
    MisalignedPointer {
        /// The field carrying the misaligned pointer.
        field: &'static str,
    },
    /// The dtype code is not one of [`super::super::SKFfiDataType`].
    UnknownDataType(i32),
    /// The descriptor declares a dimension count outside `1..=2`.
    InvalidNumberOfDimensions(i32),
    /// A dimension is negative.
    NegativeDimension(i64),
    /// A dimension is zero; every supported operation needs observations.
    ZeroDimension(usize),
    /// A negative element count.
    NegativeLength(i64),
    /// The declared element count overflows the addressable range.
    LengthOutOfRange,
    /// The element count disagrees with the shape.
    LengthMismatch {
        /// The product of the shape.
        expected: i64,
        /// The declared `len`.
        found: i64,
    },
    /// Non-null strides describe a layout other than row-major contiguous.
    NonContiguousStrides,
    /// The descriptor dtype is not the dtype the operation requires.
    DtypeMismatch {
        /// The dtype the operation requires.
        expected: &'static str,
        /// The dtype the descriptor carries.
        found: &'static str,
    },
    /// A shape describes a vector where a matrix is required, or vice versa.
    WrongRank {
        /// The rank the operation requires.
        expected: i32,
        /// The rank the descriptor carries.
        found: i32,
    },
}

impl SKFfiArrayError {
    /// The stable status this failure maps to.
    pub fn status(&self) -> crate::SKFfiStatus {
        match self {
            SKFfiArrayError::NonContiguousStrides => crate::SKFfiStatus::Unsupported,
            _ => crate::SKFfiStatus::InvalidArgument,
        }
    }
}

impl fmt::Display for SKFfiArrayError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SKFfiArrayError::NullData => {
                formatter.write_str("array descriptor has a null data pointer")
            }
            SKFfiArrayError::NullShape => {
                formatter.write_str("array descriptor has a null shape pointer")
            }
            SKFfiArrayError::MisalignedPointer { field } => {
                write!(formatter, "array descriptor field `{field}` is not aligned")
            }
            SKFfiArrayError::UnknownDataType(code) => {
                write!(formatter, "unknown array dtype code {code}")
            }
            SKFfiArrayError::InvalidNumberOfDimensions(ndim) => {
                write!(
                    formatter,
                    "array descriptor declares ndim {ndim}; expected 1 or 2"
                )
            }
            SKFfiArrayError::NegativeDimension(dimension) => {
                write!(formatter, "array dimension {dimension} is negative")
            }
            SKFfiArrayError::ZeroDimension(dimension) => {
                write!(formatter, "array dimension {dimension} is zero")
            }
            SKFfiArrayError::NegativeLength(len) => {
                write!(formatter, "array len {len} is negative")
            }
            SKFfiArrayError::LengthOutOfRange => {
                formatter.write_str("array element count exceeds the addressable range")
            }
            SKFfiArrayError::LengthMismatch { expected, found } => write!(
                formatter,
                "array length mismatch: expected {expected} elements from the shape, found {found}"
            ),
            SKFfiArrayError::NonContiguousStrides => formatter.write_str(
                "strided arrays are not supported; pass strides = NULL for row-major data",
            ),
            SKFfiArrayError::DtypeMismatch { expected, found } => {
                write!(
                    formatter,
                    "array dtype mismatch: expected {expected}, found {found}"
                )
            }
            SKFfiArrayError::WrongRank { expected, found } => {
                write!(
                    formatter,
                    "array rank mismatch: expected {expected}, found {found}"
                )
            }
        }
    }
}
