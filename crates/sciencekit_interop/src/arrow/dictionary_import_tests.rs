//! TDD suite for dictionary and text import (task 1.8).

use std::sync::Arc;

use arrow_array::{Array, DictionaryArray, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema};

use super::import::{
    sk_dictionary_label_table, sk_label_tables_from_record_batch,
    sk_tabular_batch_from_record_batch, sk_text_borrowed_view,
};
use crate::columns::SKColumnData;
use crate::missing::SKMissingValuePolicy;

fn dictionary_batch() -> RecordBatch {
    let dictionary = DictionaryArray::<arrow_array::types::Int32Type>::from_iter(vec![
        Some("cat"),
        Some("fish"),
        Some("dog"),
        None,
        Some("cat"),
    ]);
    RecordBatch::try_new(
        Arc::new(Schema::new(vec![Field::new(
            "species",
            dictionary.data_type().clone(),
            true,
        )])),
        vec![Arc::new(dictionary)],
    )
    .expect("a dictionary batch")
}

#[test]
fn sk_dictionary_imports_codes_and_a_deterministic_label_table() {
    let batch = dictionary_batch();
    let tabular = sk_tabular_batch_from_record_batch(
        &batch,
        SKMissingValuePolicy::CategoricalMissingCode,
        0,
        true,
    )
    .expect("a dictionary import");

    assert_eq!(
        tabular.column_named("species"),
        Some(&SKColumnData::CategoryCode(vec![
            Some(0),
            Some(1),
            Some(2),
            None,
            Some(0)
        ]))
    );

    let table = sk_dictionary_label_table(batch.column(0).as_ref())
        .expect("a dictionary")
        .expect("a label table");
    let labels: Vec<&str> = table.labels().collect();
    assert_eq!(labels, vec!["cat", "fish", "dog"]);
    assert_eq!(table.number_of_classes(), 3);
    assert_eq!(table.label_of(1), Some("fish"));
    assert_eq!(table.index_of("dog"), Some(2));
}

#[test]
fn sk_record_batch_exposes_one_label_table_per_column() {
    let batch = dictionary_batch();
    let tables = sk_label_tables_from_record_batch(&batch).expect("tables");

    assert_eq!(tables.len(), 1);
    assert!(tables[0].is_some());
    assert!(
        sk_label_tables_from_record_batch(&individual_values_batch()).expect("tables")[0].is_none()
    );
}

fn individual_values_batch() -> RecordBatch {
    RecordBatch::try_new(
        Arc::new(Schema::new(vec![Field::new("name", DataType::Utf8, false)])),
        vec![Arc::new(StringArray::from(vec!["a", "b"]))],
    )
    .expect("a text batch")
}

#[test]
fn sk_dictionary_label_table_is_absent_for_plain_text() {
    assert!(
        sk_dictionary_label_table(individual_values_batch().column(0).as_ref())
            .expect("a plain text column")
            .is_none()
    );
}

#[test]
fn sk_text_import_owns_the_values_but_the_borrowed_view_aliases_the_text() {
    let batch = individual_values_batch();
    let tabular = sk_tabular_batch_from_record_batch(&batch, SKMissingValuePolicy::Reject, 0, true)
        .expect("a text import");
    assert_eq!(
        tabular.column_named("name"),
        Some(&SKColumnData::Nominal(vec![
            Some("a".to_string()),
            Some("b".to_string())
        ]))
    );

    let strings = batch
        .column(0)
        .as_any()
        .downcast_ref::<StringArray>()
        .expect("a text column");
    let borrowed = sk_text_borrowed_view(batch.column(0).as_ref()).expect("a borrowed view");
    assert_eq!(borrowed.number_of_rows(), 2);
    assert_eq!(
        borrowed.value(0).expect("a value").as_ptr(),
        strings.value(0).as_ptr()
    );
    assert_eq!(borrowed.value(1), Some("b"));
    assert_eq!(borrowed.as_str_view().expect("a text view").len(), 2);
}

#[test]
fn sk_dictionary_null_under_reject_names_the_column() {
    let dictionary =
        DictionaryArray::<arrow_array::types::Int32Type>::from_iter(vec![Some("cat"), None]);
    let batch = RecordBatch::try_new(
        Arc::new(Schema::new(vec![Field::new(
            "species",
            dictionary.data_type().clone(),
            true,
        )])),
        vec![Arc::new(dictionary)],
    )
    .expect("a dictionary batch");

    let error = sk_tabular_batch_from_record_batch(&batch, SKMissingValuePolicy::Reject, 0, true)
        .expect_err("the reject policy fails");
    assert!(error.to_string().contains("species"), "{error}");
}

#[test]
fn sk_dictionary_codes_are_stable_across_repeated_imports() {
    let first = sk_tabular_batch_from_record_batch(
        &dictionary_batch(),
        SKMissingValuePolicy::CategoricalMissingCode,
        0,
        true,
    )
    .expect("a first import");
    let second = sk_tabular_batch_from_record_batch(
        &dictionary_batch(),
        SKMissingValuePolicy::CategoricalMissingCode,
        0,
        true,
    )
    .expect("a second import");
    assert_eq!(first, second);
}
