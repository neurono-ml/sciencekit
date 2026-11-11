//! TDD suite for dictionary value-slot remapping, empty string dictionaries,
//! non-string dictionary rejection and null-slot errors (finding B5).

use std::sync::Arc;

use arrow_array::types::Int32Type;
use arrow_array::{Array, DictionaryArray, Int32Array, Int64Array, RecordBatch, StringArray};
use arrow_schema::{Field, Schema};
use sciencekit_common::SKError;

use super::import::{
    sk_column_data_from_arrow_array, sk_dictionary_label_table, sk_label_tables_from_record_batch,
    sk_tabular_batch_from_record_batch,
};
use crate::columns::SKColumnData;
use crate::missing::SKMissingValuePolicy;

#[test]
fn sk_duplicate_and_null_dictionary_values_remap_keys_reversibly() {
    // Values carry an unreferenced null slot (0) and duplicate "cat" slots (1, 3).
    let values = Arc::new(StringArray::from(vec![
        None,
        Some("cat"),
        Some("dog"),
        Some("cat"),
    ]));
    let keys = Int32Array::from(vec![Some(3), Some(2), Some(1), None]);
    let dictionary = DictionaryArray::<Int32Type>::try_new(keys, values).expect("a dictionary");
    let batch = RecordBatch::try_new(
        Arc::new(Schema::new(vec![Field::new(
            "pet",
            dictionary.data_type().clone(),
            true,
        )])),
        vec![Arc::new(dictionary)],
    )
    .expect("a dictionary batch");

    let tabular = sk_tabular_batch_from_record_batch(
        &batch,
        SKMissingValuePolicy::CategoricalMissingCode,
        0,
        true,
    )
    .expect("a remapped dictionary import");
    let table = sk_dictionary_label_table(batch.column(0).as_ref())
        .expect("a valid dictionary")
        .expect("a label table");

    let Some(SKColumnData::CategoryCode(codes)) = tabular.column_named("pet") else {
        panic!("expected category codes");
    };
    assert_eq!(codes, &vec![Some(0), Some(1), Some(0), None]);
    let restored: Vec<Option<&str>> = codes
        .iter()
        .map(|code| code.map(|value| table.label_of(value as usize).expect("mapped label")))
        .collect();
    assert_eq!(restored, vec![Some("cat"), Some("dog"), Some("cat"), None]);
}

#[test]
fn sk_dictionary_key_referencing_null_dictionary_slot_is_rejected() {
    let values = Arc::new(StringArray::from(vec![None, Some("cat")]));
    let keys = Int32Array::from(vec![Some(0), Some(1)]);
    let dictionary = DictionaryArray::<Int32Type>::try_new(keys, values).expect("a dictionary");

    let error = sk_column_data_from_arrow_array(
        "pet",
        &dictionary,
        SKMissingValuePolicy::CategoricalMissingCode,
    )
    .expect_err("a key pointing to a null dictionary value slot is invalid");
    assert!(matches!(error, SKError::Conversion(_)), "{error}");
    assert!(error.to_string().contains("pet"), "{error}");
}

#[test]
fn sk_empty_string_dictionary_yields_an_empty_label_table() {
    let empty_dictionary = DictionaryArray::<Int32Type>::try_new(
        Int32Array::from(Vec::<i32>::new()),
        Arc::new(StringArray::from(Vec::<&str>::new())),
    )
    .expect("an empty dictionary");
    let table = sk_dictionary_label_table(&empty_dictionary)
        .expect("empty string dictionary is valid")
        .expect("a label table");
    assert_eq!(table.number_of_classes(), 0);

    let batch = RecordBatch::try_new(
        Arc::new(Schema::new(vec![Field::new(
            "empty_species",
            empty_dictionary.data_type().clone(),
            true,
        )])),
        vec![Arc::new(empty_dictionary)],
    )
    .expect("an empty dictionary batch");
    let tabular = sk_tabular_batch_from_record_batch(&batch, SKMissingValuePolicy::Reject, 0, true)
        .expect("empty dictionary batch imports");
    assert_eq!(tabular.number_of_rows(), 0);
}

#[test]
fn sk_non_string_dictionary_fails_with_unsupported_column_type_and_field_name() {
    let values = Arc::new(Int64Array::from(vec![100_i64, 200]));
    let keys = Int32Array::from(vec![0, 1]);
    let numeric_dictionary =
        DictionaryArray::<Int32Type>::try_new(keys, values).expect("a numeric dictionary");

    let column_error = sk_column_data_from_arrow_array(
        "bucket_id",
        &numeric_dictionary,
        SKMissingValuePolicy::Reject,
    )
    .expect_err("non-string dictionary must fail");
    match column_error {
        SKError::UnsupportedColumnType { column, .. } => assert_eq!(column, "bucket_id"),
        other => panic!("expected UnsupportedColumnType, got {other:?}"),
    }

    let table_error =
        sk_dictionary_label_table(&numeric_dictionary).expect_err("non-string dictionary fails");
    assert!(
        matches!(table_error, SKError::UnsupportedColumnType { .. }),
        "{table_error}"
    );

    let batch = RecordBatch::try_new(
        Arc::new(Schema::new(vec![Field::new(
            "bucket_id",
            numeric_dictionary.data_type().clone(),
            false,
        )])),
        vec![Arc::new(numeric_dictionary)],
    )
    .expect("a batch");
    let batch_error = sk_label_tables_from_record_batch(&batch)
        .expect_err("record-batch label tables reject non-string dictionary with real field name");
    match batch_error {
        SKError::UnsupportedColumnType { column, .. } => assert_eq!(column, "bucket_id"),
        other => panic!("expected UnsupportedColumnType naming `bucket_id`, got {other:?}"),
    }
}

#[test]
fn sk_dictionary_target_extracts_and_restores_original_labels() {
    let targets = DictionaryArray::<Int32Type>::from_iter(vec![
        Some("cat"),
        Some("fish"),
        Some("dog"),
        Some("cat"),
    ]);
    let batch = RecordBatch::try_new(
        Arc::new(Schema::new(vec![
            Field::new("feature", arrow_schema::DataType::Float64, false),
            Field::new("species", targets.data_type().clone(), false),
        ])),
        vec![
            Arc::new(arrow_array::Float64Array::from(vec![1.0, 2.0, 3.0, 4.0])),
            Arc::new(targets),
        ],
    )
    .expect("a supervised dictionary batch");

    let tabular = sk_tabular_batch_from_record_batch(&batch, SKMissingValuePolicy::Reject, 0, true)
        .expect("a dictionary import");
    let table = sk_dictionary_label_table(batch.column(1).as_ref())
        .expect("valid dictionary")
        .expect("a label table");
    let selection = crate::selection::SKColumnSelection::builder()
        .feature("feature")
        .target("species")
        .build()
        .expect("a supervised selection");
    let extracted = tabular
        .extract_target::<f64>(&selection, SKMissingValuePolicy::Reject)
        .expect("extracted dictionary target");
    let crate::batch::SKTargetExtraction::Integer(codes) = extracted else {
        panic!("expected integer target codes, got {extracted:?}");
    };
    let restored: Vec<&str> = codes
        .iter()
        .map(|&code| table.label_of(code as usize).expect("reversible label"))
        .collect();
    assert_eq!(restored, vec!["cat", "fish", "dog", "cat"]);
}
