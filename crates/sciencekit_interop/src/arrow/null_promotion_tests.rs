//! TDD suite for nullable integer preservation at import and scalar precision
//! guards at materialization (finding B4).

use std::sync::Arc;

use arrow_array::{Int64Array, RecordBatch};
use arrow_buffer::NullBuffer;
use arrow_schema::{DataType, Field, Schema};
use sciencekit_common::SKError;

use super::import::sk_tabular_batch_from_record_batch;
use crate::columns::SKColumnData;
use crate::missing::SKMissingValuePolicy;
use crate::schema::SKLogicalType;
use crate::selection::SKColumnSelection;

fn nullable_integer_batch(values: Vec<i64>) -> RecordBatch {
    let schema = Schema::new(vec![Field::new("count", DataType::Int64, true)]);
    RecordBatch::try_new(
        Arc::new(schema),
        vec![Arc::new(Int64Array::new(
            values.into(),
            Some(NullBuffer::from(vec![true, false, true])),
        ))],
    )
    .expect("a nullable integer batch")
}

#[test]
fn sk_nullable_integer_keeps_schema_and_promotes_at_materialization() {
    let with_nulls = sk_tabular_batch_from_record_batch(
        &nullable_integer_batch(vec![1, 2, 3]),
        SKMissingValuePolicy::NumericNaN,
        0,
        false,
    )
    .expect("nullable integer import");
    let complete_arrow = RecordBatch::try_new(
        Arc::new(Schema::new(vec![Field::new(
            "count",
            DataType::Int64,
            true,
        )])),
        vec![Arc::new(Int64Array::from(vec![4_i64, 5]))],
    )
    .expect("a complete integer batch");
    let without_nulls = sk_tabular_batch_from_record_batch(
        &complete_arrow,
        SKMissingValuePolicy::NumericNaN,
        1,
        true,
    )
    .expect("complete integer import");

    assert_eq!(with_nulls.schema(), without_nulls.schema());
    assert_eq!(
        with_nulls
            .schema()
            .column("count")
            .expect("column")
            .logical_type(),
        SKLogicalType::Integer64
    );
    assert_eq!(
        with_nulls.column_named("count"),
        Some(&SKColumnData::Integer64(vec![Some(1), None, Some(3)]))
    );

    let selection = SKColumnSelection::builder()
        .feature("count")
        .build()
        .expect("a selection");
    let matrix = with_nulls
        .to_dense::<f64>(&selection, SKMissingValuePolicy::NumericNaN)
        .expect("materialization promotes to f64");
    assert_eq!(matrix[[0, 0]], 1.0);
    assert!(matrix[[1, 0]].is_nan());
    assert_eq!(matrix[[2, 0]], 3.0);
}

#[test]
fn sk_lossy_integer_null_promotion_is_rejected_for_f64_and_f32() {
    let selection = SKColumnSelection::builder()
        .feature("count")
        .build()
        .expect("a selection");

    let oversized_f64 = sk_tabular_batch_from_record_batch(
        &nullable_integer_batch(vec![(1_i64 << 60) + 1, 0, 1]),
        SKMissingValuePolicy::NumericNaN,
        0,
        true,
    )
    .expect("import preserves i64");
    let f64_error = oversized_f64
        .to_dense::<f64>(&selection, SKMissingValuePolicy::NumericNaN)
        .expect_err("f64 mantissa cannot represent 2^60 + 1");
    assert!(f64_error.to_string().contains("count"), "{f64_error}");

    let exact_in_f64_lossy_in_f32 = 16_777_217_i64;
    let boundary_batch = sk_tabular_batch_from_record_batch(
        &nullable_integer_batch(vec![exact_in_f64_lossy_in_f32, 0, 1]),
        SKMissingValuePolicy::NumericNaN,
        0,
        true,
    )
    .expect("import preserves i64");
    let promoted_f64 = boundary_batch
        .to_dense::<f64>(&selection, SKMissingValuePolicy::NumericNaN)
        .expect("2^24 + 1 is exact in f64");
    assert_eq!(promoted_f64[[0, 0]], 16_777_217.0_f64);
    assert!(promoted_f64[[1, 0]].is_nan());

    let f32_error = boundary_batch
        .to_dense::<f32>(&selection, SKMissingValuePolicy::NumericNaN)
        .expect_err("2^24 + 1 truncates in f32 and must be rejected");
    assert!(matches!(f32_error, SKError::Conversion(_)), "{f32_error}");
}
