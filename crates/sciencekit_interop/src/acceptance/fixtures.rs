//! Shared fixtures for the acceptance suite.

use crate::batch::SKTabularBatch;
use crate::columns::SKColumnData;
use crate::schema::{SKColumnRole, SKColumnSchema, SKLogicalType, SKTabularSchema};
use crate::selection::SKColumnSelection;

/// A five-column table mixing every supported numeric type and a nominal
/// column that is ignored by the numeric selection.
pub(crate) fn numeric_schema() -> SKTabularSchema {
    SKTabularSchema::new(vec![
        SKColumnSchema::new("single", SKLogicalType::Float32, SKColumnRole::Feature),
        SKColumnSchema::new("double", SKLogicalType::Float64, SKColumnRole::Feature),
        SKColumnSchema::new("count", SKLogicalType::Integer64, SKColumnRole::Feature),
        SKColumnSchema::new("flag", SKLogicalType::Boolean, SKColumnRole::Feature),
        SKColumnSchema::new("species", SKLogicalType::Nominal, SKColumnRole::Ignored),
    ])
    .unwrap()
}

/// The ordered selection of the four numeric columns.
pub(crate) fn numeric_selection() -> SKColumnSelection {
    SKColumnSelection::builder()
        .feature("single")
        .feature("double")
        .feature("count")
        .feature("flag")
        .build()
        .unwrap()
}

/// Build a synthetic table of `rows` rows with predictable values.
pub(crate) fn numeric_batch(rows: usize) -> SKTabularBatch {
    let singles: Vec<Option<f32>> = (0..rows).map(|row| Some(row as f32 * 0.5)).collect();
    let doubles: Vec<Option<f64>> = (0..rows).map(|row| Some(row as f64)).collect();
    let counts: Vec<Option<i64>> = (0..rows).map(|row| Some(row as i64)).collect();
    let flags: Vec<Option<bool>> = (0..rows).map(|row| Some(row % 2 == 0)).collect();
    let species: Vec<Option<String>> = (0..rows)
        .map(|row| Some(format!("species_{}", row % 3)))
        .collect();
    SKTabularBatch::new(
        numeric_schema(),
        vec![
            SKColumnData::Float32(singles),
            SKColumnData::Float64(doubles),
            SKColumnData::Integer64(counts),
            SKColumnData::Boolean(flags),
            SKColumnData::Nominal(species),
        ],
        0,
        true,
    )
    .unwrap()
}
