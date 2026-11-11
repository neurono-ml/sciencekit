//! TDD suite for Arrow null materialization under each missing-value policy
//! (task 1.6).

use std::sync::Arc;

use arrow_array::types::Int32Type;
use arrow_array::{Array, DictionaryArray, Float32Array, Float64Array, RecordBatch, StringArray};
use arrow_buffer::NullBuffer;
use arrow_schema::{DataType, Field, Schema};
use sciencekit_common::SK_CATEGORICAL_MISSING_CODE;

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
    assert!(message.contains("value"), "{message}");
    assert!(message.contains("reject"), "{message}");
}

#[test]
fn sk_nullable_string_and_dictionary_extract_categorical_missing_code() {
    let dictionary = DictionaryArray::<Int32Type>::from_iter(vec![Some("x"), None, Some("z")]);
    let schema = Schema::new(vec![
        Field::new("species", DataType::Utf8, true),
        Field::new("breed", dictionary.data_type().clone(), true),
    ]);
    let batch = RecordBatch::try_new(
        Arc::new(schema),
        vec![
            Arc::new(StringArray::from(vec![Some("a"), None, Some("c")])),
            Arc::new(dictionary),
        ],
    )
    .expect("a nullable categorical batch");
    let tabular = sk_tabular_batch_from_record_batch(
        &batch,
        SKMissingValuePolicy::CategoricalMissingCode,
        0,
        true,
    )
    .expect("a categorical import");
    let selection = SKColumnSelection::builder()
        .feature("species")
        .feature("breed")
        .build()
        .expect("a categorical selection");
    let extracted = tabular
        .extract_categorical(&selection, SKMissingValuePolicy::CategoricalMissingCode)
        .expect("categorical extraction");
    let codes = extracted.codes().expect("integer codes");
    assert_eq!(codes[[1, 0]], SK_CATEGORICAL_MISSING_CODE);
    assert_eq!(codes[[1, 1]], SK_CATEGORICAL_MISSING_CODE);
    for column in [0, 1] {
        for row in [0, 2] {
            assert!(codes[[row, column]] >= 0);
            assert_ne!(codes[[row, column]], SK_CATEGORICAL_MISSING_CODE);
        }
    }
}

#[test]
fn sk_non_finite_or_overflowing_constant_fill_is_rejected() {
    for bad_fill in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let error = sk_tabular_batch_from_record_batch(
            &nullable_f64_batch(),
            SKMissingValuePolicy::ConstantFill(bad_fill),
            0,
            true,
        )
        .expect_err("non-finite fill must be rejected");
        assert!(error.to_string().contains("constant_fill_value"), "{error}");
    }

    let f32_batch = RecordBatch::try_new(
        Arc::new(Schema::new(vec![Field::new(
            "single",
            DataType::Float32,
            true,
        )])),
        vec![Arc::new(Float32Array::new(
            vec![1.0_f32, 2.0].into(),
            Some(NullBuffer::from(vec![true, false])),
        ))],
    )
    .expect("a nullable f32 batch");
    let error = sk_tabular_batch_from_record_batch(
        &f32_batch,
        SKMissingValuePolicy::ConstantFill(1e300),
        0,
        true,
    )
    .expect_err("the fill overflows f32");
    assert!(error.to_string().contains("constant_fill_value"), "{error}");
}
