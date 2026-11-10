//! TDD suite for Arrow null materialization under each missing-value policy
//! (task 1.6).

use std::sync::Arc;

use arrow_array::{Float64Array, Int64Array, RecordBatch, StringArray};
use arrow_buffer::NullBuffer;
use arrow_schema::{DataType, Field, Schema};

use super::import::sk_tabular_batch_from_record_batch;
use crate::columns::SKColumnData;
use crate::missing::SKMissingValuePolicy;
use crate::selection::SKColumnSelection;

fn nullable_f64_batch() -> RecordBatch {
    let schema = Schema::new(vec![Field::new("value", DataType::Float64, true)]);
    RecordBatch::try_new(
        Arc::new(schema),
        vec![Arc::new(Float64Array::new(
            vec![1.0, 2.0, 3.0].into(),
            Some(NullBuffer::from(vec![true, false, true])),
        ))],
    )
    .expect("a nullable batch")
}

#[test]
fn sk_numeric_nan_policy_writes_nan_at_null_positions() {
    let tabular = sk_tabular_batch_from_record_batch(
        &nullable_f64_batch(),
        SKMissingValuePolicy::NumericNaN,
        0,
        true,
    )
    .expect("a policy-managed import");

    let column = tabular.column_named("value").expect("the column");
    assert_eq!(column.null_count(), 0, "the null is materialized");
    match column {
        SKColumnData::Float64(values) => {
            assert_eq!(values[0], Some(1.0));
            assert!(values[1].expect("a value").is_nan());
            assert_eq!(values[2], Some(3.0));
        }
        other => panic!("unexpected column: {other:?}"),
    }
}

#[test]
fn sk_constant_fill_policy_writes_the_constant_at_null_positions() {
    let policy = SKMissingValuePolicy::constant_fill(9.5).expect("a finite fill");
    let tabular = sk_tabular_batch_from_record_batch(&nullable_f64_batch(), policy, 0, true)
        .expect("a policy-managed import");

    assert_eq!(
        tabular.column_named("value"),
        Some(&SKColumnData::Float64(vec![
            Some(1.0),
            Some(9.5),
            Some(3.0)
        ]))
    );
}

#[test]
fn sk_reject_policy_reports_the_arrow_field() {
    let error = sk_tabular_batch_from_record_batch(
        &nullable_f64_batch(),
        SKMissingValuePolicy::Reject,
        0,
        true,
    )
    .expect_err("the reject policy fails on a null");

    let message = error.to_string();
    assert!(
        message.contains("value"),
        "the error names the field: {message}"
    );
    assert!(
        message.contains("reject"),
        "the error names the policy: {message}"
    );
}

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
fn sk_nullable_integer_promotes_to_float64_for_the_nan_policy() {
    let tabular = sk_tabular_batch_from_record_batch(
        &nullable_integer_batch(vec![1, 2, 3]),
        SKMissingValuePolicy::NumericNaN,
        0,
        true,
    )
    .expect("a promoted import");

    let column = tabular.column_named("count").expect("the column");
    match column {
        SKColumnData::Float64(values) => {
            assert_eq!(values[0], Some(1.0));
            assert!(values[1].expect("a value").is_nan());
            assert_eq!(values[2], Some(3.0));
        }
        other => panic!("unexpected promoted column: {other:?}"),
    }
    let selection = SKColumnSelection::builder()
        .feature("count")
        .build()
        .expect("a selection");
    let matrix = tabular
        .to_dense::<f64>(&selection, SKMissingValuePolicy::NumericNaN)
        .expect("the promoted column materializes");
    assert!(matrix[[1, 0]].is_nan());
}

#[test]
fn sk_lossy_integer_promotion_is_rejected() {
    let oversized = (1_i64 << 60) + 1;
    let error = sk_tabular_batch_from_record_batch(
        &nullable_integer_batch(vec![oversized, 0, 1]),
        SKMissingValuePolicy::NumericNaN,
        0,
        true,
    )
    .expect_err("the promotion would truncate");

    let message = error.to_string();
    assert!(message.contains("count"), "{message}");
    assert!(message.contains("exactly"), "{message}");
}

#[test]
fn sk_categorical_missing_code_is_preserved_for_the_categorical_path() {
    let schema = Schema::new(vec![Field::new("species", DataType::Utf8, true)]);
    let batch = RecordBatch::try_new(
        Arc::new(schema),
        vec![Arc::new(StringArray::from(vec![
            Some("a"),
            None,
            Some("c"),
        ]))],
    )
    .expect("a nullable text batch");

    let tabular = sk_tabular_batch_from_record_batch(
        &batch,
        SKMissingValuePolicy::CategoricalMissingCode,
        0,
        true,
    )
    .expect("a categorical import");

    assert_eq!(
        tabular.column_named("species"),
        Some(&SKColumnData::Nominal(vec![
            Some("a".to_string()),
            None,
            Some("c".to_string())
        ]))
    );
}

#[test]
fn sk_non_finite_fill_overflowing_f32_is_rejected() {
    let schema = Schema::new(vec![Field::new("single", DataType::Float32, true)]);
    let batch = RecordBatch::try_new(
        Arc::new(schema),
        vec![Arc::new(arrow_array::Float32Array::new(
            vec![1.0_f32, 2.0].into(),
            Some(NullBuffer::from(vec![true, false])),
        ))],
    )
    .expect("a nullable float batch");

    let error = sk_tabular_batch_from_record_batch(
        &batch,
        SKMissingValuePolicy::ConstantFill(1e300),
        0,
        true,
    )
    .expect_err("the fill overflows f32");

    assert!(error.to_string().contains("constant_fill_value"), "{error}");
}
