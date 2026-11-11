//! Zero-copy views over Arrow primitive buffers, and the conversion report
//! (spec `arrow-columnar-interop`).
//!
//! Arrow stores primitive columns contiguously, so a complete `f32`, `f64` or
//! `i64` column is the same memory a sciencekit numeric column view needs. The
//! borrowed view below aliases that buffer with no copy and no `unsafe`: it
//! holds a slice borrowed from the Arrow array, which keeps the buffer alive
//! for as long as the view lives.

use std::any::TypeId;

use arrow_array::types::{Float32Type, Float64Type, Int64Type};
use arrow_array::{Array, PrimitiveArray};
use arrow_schema::DataType;
use sciencekit_common::{SKError, SKFloat};

use crate::views::SKNumericColumnView;

/// A borrowed numeric column whose bytes are the Arrow buffer itself.
#[derive(Debug, Clone, Copy)]
pub enum SKArrowBorrowedColumn<'a> {
    /// A borrowed single-precision column.
    Float32(&'a [f32]),
    /// A borrowed double-precision column.
    Float64(&'a [f64]),
    /// A borrowed signed 64-bit integer column.
    Integer64(&'a [i64]),
}

#[allow(clippy::needless_lifetimes)]
impl<'a> SKArrowBorrowedColumn<'a> {
    /// The number of rows in the borrowed column.
    pub fn number_of_rows(&self) -> usize {
        match self {
            SKArrowBorrowedColumn::Float32(values) => values.len(),
            SKArrowBorrowedColumn::Float64(values) => values.len(),
            SKArrowBorrowedColumn::Integer64(values) => values.len(),
        }
    }

    /// The borrowed values, when the column is single-precision.
    pub fn as_f32_column_view(&self) -> Option<SKNumericColumnView<'a, f32>> {
        match self {
            SKArrowBorrowedColumn::Float32(values) => Some(SKNumericColumnView::new(values)),
            _ => None,
        }
    }

    /// The borrowed values, when the column is double-precision.
    pub fn as_f64_column_view(&self) -> Option<SKNumericColumnView<'a, f64>> {
        match self {
            SKArrowBorrowedColumn::Float64(values) => Some(SKNumericColumnView::new(values)),
            _ => None,
        }
    }

    /// The borrowed values, when the column is signed 64-bit.
    pub fn as_i64_column_view(&self) -> Option<SKNumericColumnView<'a, i64>> {
        match self {
            SKArrowBorrowedColumn::Integer64(values) => Some(SKNumericColumnView::new(values)),
            _ => None,
        }
    }
}

/// Borrow an Arrow primitive column without copying.
///
/// # Errors
/// Returns [`SKError::UnsupportedRepresentation`] when the column carries
/// nulls, is not one of `Float32`/`Float64`/`Int64`, or cannot be downcast, so
/// the caller materializes through the normal import path instead.
pub fn sk_arrow_borrowed_column(array: &dyn Array) -> Result<SKArrowBorrowedColumn<'_>, SKError> {
    if array.null_count() > 0 {
        return Err(SKError::UnsupportedRepresentation {
            representation: "arrow_nullable",
            suggestion: "materialize the column through the record-batch import so the missing-value policy is applied",
        });
    }
    match array.data_type() {
        DataType::Float32 => array
            .as_any()
            .downcast_ref::<PrimitiveArray<Float32Type>>()
            .map(|values| SKArrowBorrowedColumn::Float32(values.values().as_ref()))
            .ok_or_else(sk_borrow_failed),
        DataType::Float64 => array
            .as_any()
            .downcast_ref::<PrimitiveArray<Float64Type>>()
            .map(|values| SKArrowBorrowedColumn::Float64(values.values().as_ref()))
            .ok_or_else(sk_borrow_failed),
        DataType::Int64 => array
            .as_any()
            .downcast_ref::<PrimitiveArray<Int64Type>>()
            .map(|values| SKArrowBorrowedColumn::Integer64(values.values().as_ref()))
            .ok_or_else(sk_borrow_failed),
        _ => Err(SKError::UnsupportedRepresentation {
            representation: "arrow_non_primitive",
            suggestion: "select a float32, float64 or int64 column to borrow directly; other types need materialization",
        }),
    }
}

fn sk_borrow_failed() -> SKError {
    SKError::Conversion(
        "the Arrow array's data type and its concrete array disagree, so it cannot be borrowed"
            .to_string(),
    )
}

/// How an Arrow column relates to the model scalar.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SKArrowConversionReport {
    zero_copy: bool,
    cast: bool,
}

impl SKArrowConversionReport {
    /// Whether the buffer can be borrowed for the model scalar unchanged.
    pub fn is_zero_copy(&self) -> bool {
        self.zero_copy
    }

    /// Whether the values must be converted to a different scalar precision.
    pub fn is_cast(&self) -> bool {
        self.cast
    }
}

/// Report whether a column can be borrowed as `F` or requires a cast.
///
/// The report is about the buffer, not about the owned import: the tabular
/// batch always owns its values, so `zero_copy` tells the caller that a
/// borrowed view of the same column is available through
/// [`sk_arrow_borrowed_column`], while `cast` tells it the Arrow dtype differs
/// from the model scalar.
pub fn sk_arrow_conversion_report<F: SKFloat>(array: &dyn Array) -> SKArrowConversionReport {
    let matches_scalar = (array.data_type() == &DataType::Float32
        && TypeId::of::<F>() == TypeId::of::<f32>())
        || (array.data_type() == &DataType::Float64 && TypeId::of::<F>() == TypeId::of::<f64>());
    let zero_copy = matches_scalar && array.null_count() == 0;
    let cast = matches!(
        array.data_type(),
        DataType::Float32
            | DataType::Float64
            | DataType::Int8
            | DataType::Int16
            | DataType::Int32
            | DataType::Int64
            | DataType::UInt8
            | DataType::UInt16
            | DataType::UInt32
    ) && !matches_scalar;
    SKArrowConversionReport { zero_copy, cast }
}
