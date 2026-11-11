//! TDD suite for boolean import as numeric features and as categorical codes
//! (task 1.7).

use std::sync::Arc;

use arrow_array::{BooleanArray, RecordBatch};
use arrow_buffer::NullBuffer;
use arrow_schema::{DataType, Field, Schema};

use super::import::{sk_boolean_category_codes, sk_tabular_batch_from_record_batch};
use crate::columns::SKColumnData;
use crate::missing::SKMissingValuePolicy;
use crate::selection::SKColumnSelection;
use sciencekit_common::SK_CATEGORICAL_MISSING_CODE;

fn boolean_batch(values: Vec<Option<bool>>) -> RecordBatch {
    RecordBatch::try_new(
        Arc::new(Schema::new(vec![Field::new(
            "flag",
            DataType::Boolean,
            true,
        )])),
        vec![Arc::new(BooleanArray::from(values))],
    )
    .expect("a boolean batch")
}

#[test]
fn sk_boolean_feature_imports_as_zero_and_one() {
    let batch = boolean_batch(vec![Some(true), Some(false), Some(true)]);
    let tabular = sk_tabular_batch_from_record_batch(&batch, SKMissingValuePolicy::Reject, 0, true)
        .expect("a boolean import");

    let selection = SKColumnSelection::builder()
        .feature("flag")
        .build()
        .expect("a selection");
    let matrix = tabular
        .to_dense::<f64>(&selection, SKMissingValuePolicy::Reject)
        .expect("a numeric materialization");
    assert_eq!(matrix, ndarray::array![[1.0], [0.0], [1.0]]);
}

#[test]
fn sk_boolean_column_becomes_category_codes_on_request() {
    let batch = boolean_batch(vec![Some(true), Some(false), Some(true)]);
    let tabular = sk_tabular_batch_from_record_batch(&batch, SKMissingValuePolicy::Reject, 0, true)
        .expect("a boolean import");
    let column = tabular.column_named("flag").expect("the column");

    let codes = sk_boolean_category_codes("flag", column, SKMissingValuePolicy::Reject)
        .expect("a categorical conversion");
    assert_eq!(
        codes,
        SKColumnData::CategoryCode(vec![Some(1), Some(0), Some(1)])
    );
}

#[test]
fn sk_boolean_null_becomes_the_missing_category_code() {
    let batch = boolean_batch(vec![Some(true), None, Some(false)]);
    let tabular = sk_tabular_batch_from_record_batch(
        &batch,
        SKMissingValuePolicy::CategoricalMissingCode,
        0,
        true,
    )
    .expect("a categorical import");
    let column = tabular.column_named("flag").expect("the column");

    let codes =
        sk_boolean_category_codes("flag", column, SKMissingValuePolicy::CategoricalMissingCode)
            .expect("a categorical conversion");
    assert_eq!(
        codes,
        SKColumnData::CategoryCode(vec![Some(1), None, Some(0)])
    );

    let selection = SKColumnSelection::builder()
        .feature("flag")
        .build()
        .expect("a selection");
    let batch = RecordBatch::try_new(
        Arc::new(Schema::new(vec![
            Field::new("flag", DataType::Int64, true).with_metadata(
                std::collections::HashMap::from([(
                    super::SK_ARROW_LOGICAL_TYPE_METADATA_KEY.to_string(),
                    "category_code".to_string(),
                )]),
            ),
        ])),
        vec![Arc::new(arrow_array::Int64Array::from(vec![
            Some(1),
            None,
            Some(0),
        ]))],
    )
    .expect("a code batch");
    let tabular = sk_tabular_batch_from_record_batch(
        &batch,
        SKMissingValuePolicy::CategoricalMissingCode,
        0,
        true,
    )
    .expect("a code import");
    let extraction = tabular
        .extract_categorical(&selection, SKMissingValuePolicy::CategoricalMissingCode)
        .expect("a categorical extraction");
    assert_eq!(
        extraction.codes().expect("codes")[[1, 0]],
        SK_CATEGORICAL_MISSING_CODE
    );
}

#[test]
fn sk_boolean_null_under_reject_is_reported() {
    let batch = boolean_batch(vec![Some(true), None]);
    let error = sk_tabular_batch_from_record_batch(&batch, SKMissingValuePolicy::Reject, 0, true)
        .expect_err("the reject policy fails");
    assert!(error.to_string().contains("flag"), "{error}");

    let tabular = sk_tabular_batch_from_record_batch(
        &boolean_batch(vec![Some(true), None]),
        SKMissingValuePolicy::CategoricalMissingCode,
        0,
        true,
    )
    .expect("a categorical import");
    let column = tabular.column_named("flag").expect("the column");
    let error = sk_boolean_category_codes("flag", column, SKMissingValuePolicy::Reject)
        .expect_err("a null cannot be coded under reject");
    assert!(error.to_string().contains("flag"), "{error}");
}

#[test]
fn sk_non_boolean_column_cannot_be_coded_as_boolean_categories() {
    let error = sk_boolean_category_codes(
        "flag",
        &SKColumnData::Float64(vec![Some(1.0)]),
        SKMissingValuePolicy::Reject,
    )
    .expect_err("a float column is not boolean");
    assert!(
        error.to_string().contains("unsupported column type"),
        "{error}"
    );
}

#[test]
fn sk_nullable_boolean_array_keeps_its_validity() {
    let array = BooleanArray::new(
        vec![true, false, true].into(),
        Some(NullBuffer::from(vec![true, false, true])),
    );
    let batch = RecordBatch::try_new(
        Arc::new(Schema::new(vec![Field::new(
            "flag",
            DataType::Boolean,
            true,
        )])),
        vec![Arc::new(array)],
    )
    .expect("a nullable boolean batch");

    let tabular = sk_tabular_batch_from_record_batch(
        &batch,
        SKMissingValuePolicy::CategoricalMissingCode,
        0,
        true,
    )
    .expect("a nullable import");
    assert_eq!(
        tabular.column_named("flag"),
        Some(&SKColumnData::Boolean(vec![Some(true), None, Some(true)]))
    );
}
