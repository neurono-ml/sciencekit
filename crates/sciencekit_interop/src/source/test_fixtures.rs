//! Shared fixtures for the source and provider tests.

use crate::batch::SKTabularBatch;
use crate::columns::SKColumnData;
use crate::schema::{SKColumnRole, SKColumnSchema, SKLogicalType, SKTabularSchema};

/// The schema used by every source fixture: `x`, `y` features and `label`.
pub(crate) fn fixture_schema() -> SKTabularSchema {
    SKTabularSchema::new(vec![
        SKColumnSchema::new("x", SKLogicalType::Float64, SKColumnRole::Feature),
        SKColumnSchema::new("y", SKLogicalType::Integer64, SKColumnRole::Feature),
        SKColumnSchema::new("label", SKLogicalType::Nominal, SKColumnRole::Target),
    ])
    .unwrap()
}

/// Build one row-aligned fixture batch carrying a single row.
pub(crate) fn fixture_batch(row: usize, is_final: bool) -> SKTabularBatch {
    let value = row as f64;
    SKTabularBatch::new(
        fixture_schema(),
        vec![
            SKColumnData::Float64(vec![Some(value)]),
            SKColumnData::Integer64(vec![Some(row as i64)]),
            SKColumnData::Nominal(vec![Some(format!("label_{row}"))]),
        ],
        row,
        is_final,
    )
    .unwrap()
}

/// Build `count` one-row fixture batches, the last one final.
pub(crate) fn fixture_batches(count: usize) -> Vec<SKTabularBatch> {
    (0..count)
        .map(|row| fixture_batch(row, row + 1 == count))
        .collect()
}
