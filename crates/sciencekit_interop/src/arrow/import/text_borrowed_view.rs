//! Borrowed text views over Arrow string columns (spec
//! `arrow-columnar-interop`).
//!
//! The text bytes stay in Arrow's offsets/data buffers; only the per-row index
//! is collected, so the view aliases the original UTF-8 payload.

use arrow_array::{Array, LargeStringArray, StringArray, StringViewArray};
use arrow_schema::DataType;
use sciencekit_common::SKError;

/// A row-indexed view over an Arrow string column whose bytes are borrowed.
#[derive(Debug, Clone)]
pub struct SKArrowTextBorrowedView<'a> {
    values: Vec<Option<&'a str>>,
}

impl<'a> SKArrowTextBorrowedView<'a> {
    /// The number of rows.
    pub fn number_of_rows(&self) -> usize {
        self.values.len()
    }

    /// The borrowed text at a zero-based row.
    pub fn value(&self, row: usize) -> Option<&'a str> {
        self.values.get(row).copied().flatten()
    }

    /// All borrowed rows in order.
    pub fn as_str_view(&self) -> Option<&[Option<&'a str>]> {
        Some(&self.values)
    }
}

/// Borrow the text of an Arrow string column without copying the payload.
///
/// # Errors
/// Returns [`SKError::UnsupportedRepresentation`] for a non-text column.
pub fn sk_text_borrowed_view<'a>(
    array: &'a dyn Array,
) -> Result<SKArrowTextBorrowedView<'a>, SKError> {
    let values = match array.data_type() {
        DataType::Utf8 => array
            .as_any()
            .downcast_ref::<StringArray>()
            .map(|array| array.iter().collect()),
        DataType::LargeUtf8 => array
            .as_any()
            .downcast_ref::<LargeStringArray>()
            .map(|array| array.iter().collect()),
        DataType::Utf8View => array
            .as_any()
            .downcast_ref::<StringViewArray>()
            .map(|array| array.iter().collect()),
        _ => None,
    };
    values
        .map(|values| SKArrowTextBorrowedView { values })
        .ok_or(SKError::UnsupportedRepresentation {
            representation: "arrow_non_text",
            suggestion: "borrow only utf8, large_utf8 or utf8_view columns as text",
        })
}
