//! Acceptance scenarios for the interop error paths.

use sciencekit_common::SKError;

use super::fixtures::numeric_batch;
use crate::batch::SKTabularBatch;
use crate::columns::SKColumnData;
use crate::missing::SKMissingValuePolicy;
use crate::schema::{SKColumnRole, SKColumnSchema, SKLogicalType, SKTabularSchema};
use crate::selection::SKColumnSelection;

/// Every failure path returns the structured error naming its column.
#[test]
fn error_paths_are_structured() {
    let batch = numeric_batch(1);
    let unsupported = SKColumnSelection::builder()
        .feature("species")
        .build()
        .unwrap();
    let error = batch
        .to_dense::<f64>(&unsupported, SKMissingValuePolicy::Reject)
        .unwrap_err();
    match error {
        SKError::UnsupportedColumnType { column, .. } => assert_eq!(column, "species"),
        other => panic!("expected an unsupported-column-type error, got {other:?}"),
    }

    let null_schema = SKTabularSchema::new(vec![SKColumnSchema::new(
        "price",
        SKLogicalType::Float64,
        SKColumnRole::Feature,
    )])
    .unwrap();
    let null_batch = SKTabularBatch::new(
        null_schema,
        vec![SKColumnData::Float64(vec![None])],
        0,
        true,
    )
    .unwrap();
    let error = null_batch
        .to_dense::<f64>(
            &SKColumnSelection::builder()
                .feature("price")
                .build()
                .unwrap(),
            SKMissingValuePolicy::Reject,
        )
        .unwrap_err();
    assert!(matches!(error, SKError::NullValueInColumn { .. }));
}
