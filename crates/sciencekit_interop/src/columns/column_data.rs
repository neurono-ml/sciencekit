//! Typed, nullable column storage for one streaming block (spec
//! `dataset-interop-core`).

use crate::schema::SKLogicalType;

/// The data of one column for the width of a streaming block.
///
/// Every variant stores one optional value per row, so nullability is explicit
/// in the type instead of encoded in a magic value. Nominal text is owned
/// because a block must survive the source advancing; category codes use
/// [`sciencekit_common::SK_CATEGORICAL_MISSING_CODE`] for missing entries after
/// the policy is applied.
#[derive(Debug, Clone, PartialEq)]
pub enum SKColumnData {
    /// Single-precision floating-point values.
    Float32(Vec<Option<f32>>),
    /// Double-precision floating-point values.
    Float64(Vec<Option<f64>>),
    /// Signed 64-bit integer values.
    Integer64(Vec<Option<i64>>),
    /// Boolean values.
    Boolean(Vec<Option<bool>>),
    /// Nominal (text) values, owned by the block.
    Nominal(Vec<Option<String>>),
    /// Integer category codes.
    CategoryCode(Vec<Option<i64>>),
}

impl SKColumnData {
    /// The number of rows carried by the column.
    pub fn len(&self) -> usize {
        match self {
            SKColumnData::Float32(values) => values.len(),
            SKColumnData::Float64(values) => values.len(),
            SKColumnData::Integer64(values) => values.len(),
            SKColumnData::Boolean(values) => values.len(),
            SKColumnData::Nominal(values) => values.len(),
            SKColumnData::CategoryCode(values) => values.len(),
        }
    }

    /// Whether the column carries no row.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The number of null entries.
    pub fn null_count(&self) -> usize {
        match self {
            SKColumnData::Float32(values) => values.iter().filter(|v| v.is_none()).count(),
            SKColumnData::Float64(values) => values.iter().filter(|v| v.is_none()).count(),
            SKColumnData::Integer64(values) => values.iter().filter(|v| v.is_none()).count(),
            SKColumnData::Boolean(values) => values.iter().filter(|v| v.is_none()).count(),
            SKColumnData::Nominal(values) => values.iter().filter(|v| v.is_none()).count(),
            SKColumnData::CategoryCode(values) => values.iter().filter(|v| v.is_none()).count(),
        }
    }

    /// Whether the entry at `row` is null; `None` when the row is out of range.
    pub fn is_null(&self, row: usize) -> Option<bool> {
        match self {
            SKColumnData::Float32(values) => values.get(row).map(Option::is_none),
            SKColumnData::Float64(values) => values.get(row).map(Option::is_none),
            SKColumnData::Integer64(values) => values.get(row).map(Option::is_none),
            SKColumnData::Boolean(values) => values.get(row).map(Option::is_none),
            SKColumnData::Nominal(values) => values.get(row).map(Option::is_none),
            SKColumnData::CategoryCode(values) => values.get(row).map(Option::is_none),
        }
    }

    /// Whether any entry is null.
    pub fn has_nulls(&self) -> bool {
        self.null_count() > 0
    }

    /// The logical type the variant carries.
    pub fn logical_type(&self) -> SKLogicalType {
        match self {
            SKColumnData::Float32(_) => SKLogicalType::Float32,
            SKColumnData::Float64(_) => SKLogicalType::Float64,
            SKColumnData::Integer64(_) => SKLogicalType::Integer64,
            SKColumnData::Boolean(_) => SKLogicalType::Boolean,
            SKColumnData::Nominal(_) => SKLogicalType::Nominal,
            SKColumnData::CategoryCode(_) => SKLogicalType::CategoryCode,
        }
    }
}
