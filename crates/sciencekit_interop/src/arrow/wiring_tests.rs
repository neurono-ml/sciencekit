//! Wiring tests for the optional `arrow` feature (task 1.1).
//!
//! These tests build real Arrow arrays and record batches through the pinned
//! dependency set, so a feature that fails to compile or link fails here first.

use arrow_array::{Array, Float64Array, Int64Array, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema};
use std::sync::Arc;

#[test]
fn sk_arrow_feature_builds_arrays_and_record_batches() {
    let schema = Schema::new(vec![
        Field::new("value", DataType::Float64, false),
        Field::new("count", DataType::Int64, false),
        Field::new("label", DataType::Utf8, false),
    ]);
    let batch = RecordBatch::try_new(
        Arc::new(schema),
        vec![
            Arc::new(Float64Array::from(vec![1.5, 2.5])),
            Arc::new(Int64Array::from(vec![3, 4])),
            Arc::new(StringArray::from(vec!["a", "b"])),
        ],
    )
    .expect("a well-formed record batch");

    assert_eq!(batch.num_rows(), 2);
    assert_eq!(batch.num_columns(), 3);
    assert_eq!(batch.column(0).len(), 2);
    assert_eq!(
        batch
            .column(0)
            .as_any()
            .downcast_ref::<Float64Array>()
            .expect("a float64 column")
            .value(1),
        2.5
    );
}
