//! TDD suite for record-batch import (task 1.3).

use std::sync::Arc;

use arrow_array::{BooleanArray, Float32Array, Float64Array, Int64Array, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema};

use super::import::sk_tabular_batch_from_record_batch;
use crate::columns::SKColumnData;
use crate::missing::SKMissingValuePolicy;
use crate::selection::SKColumnSelection;

fn sample_batch() -> RecordBatch {
    let schema = Schema::new(vec![
        Field::new("single", DataType::Float32, false),
        Field::new("double", DataType::Float64, false),
        Field::new("count", DataType::Int64, false),
        Field::new("flag", DataType::Boolean, false),
        Field::new("species", DataType::Utf8, false),
    ]);
    RecordBatch::try_new(
        Arc::new(schema),
        vec![
            Arc::new(Float32Array::from(vec![1.5, 2.5, 3.5, 4.5])),
            Arc::new(Float64Array::from(vec![10.0, 20.0, 30.0, 40.0])),
            Arc::new(Int64Array::from(vec![1, 2, 3, 4])),
            Arc::new(BooleanArray::from(vec![true, false, true, false])),
            Arc::new(StringArray::from(vec!["a", "b", "c", "d"])),
        ],
    )
    .expect("a well-formed batch")
}

#[test]
fn sk_record_batch_becomes_a_tabular_batch_preserving_names_order_and_values() {
    let batch = sample_batch();
    let tabular = sk_tabular_batch_from_record_batch(&batch, SKMissingValuePolicy::Reject, 0, true)
        .expect("a supported batch");

    let names: Vec<&str> = tabular
        .schema()
        .columns()
        .iter()
        .map(|column| column.name())
        .collect();
    assert_eq!(names, vec!["single", "double", "count", "flag", "species"]);
    assert_eq!(tabular.number_of_rows(), 4);
    assert_eq!(tabular.number_of_columns(), 5);
    assert_eq!(tabular.position(), 0);
    assert!(tabular.is_final());
    assert_eq!(
        tabular.column_named("double"),
        Some(&SKColumnData::Float64(vec![
            Some(10.0),
            Some(20.0),
            Some(30.0),
            Some(40.0)
        ]))
    );
    assert_eq!(
        tabular.column_named("single"),
        Some(&SKColumnData::Float32(vec![
            Some(1.5),
            Some(2.5),
            Some(3.5),
            Some(4.5)
        ]))
    );
    assert_eq!(
        tabular.column_named("species"),
        Some(&SKColumnData::Nominal(vec![
            Some("a".to_string()),
            Some("b".to_string()),
            Some("c".to_string()),
            Some("d".to_string())
        ]))
    );

    let selection = SKColumnSelection::builder()
        .feature("single")
        .feature("double")
        .feature("count")
        .feature("flag")
        .build()
        .expect("a numeric selection");
    let matrix = tabular
        .to_dense::<f64>(&selection, SKMissingValuePolicy::Reject)
        .expect("numeric materialization");
    assert_eq!(
        matrix,
        ndarray::array![
            [1.5, 10.0, 1.0, 1.0],
            [2.5, 20.0, 2.0, 0.0],
            [3.5, 30.0, 3.0, 1.0],
            [4.5, 40.0, 4.0, 0.0]
        ]
    );
}

#[test]
fn sk_empty_record_batch_imports_with_zero_rows() {
    let schema = Schema::new(vec![
        Field::new("double", DataType::Float64, true),
        Field::new("species", DataType::Utf8, true),
    ]);
    let empty = RecordBatch::try_new(
        Arc::new(schema),
        vec![
            Arc::new(Float64Array::from(Vec::<f64>::new())),
            Arc::new(StringArray::from(Vec::<String>::new())),
        ],
    )
    .expect("an empty batch");

    let tabular = sk_tabular_batch_from_record_batch(&empty, SKMissingValuePolicy::Reject, 7, true)
        .expect("an empty import is valid");

    assert_eq!(tabular.number_of_rows(), 0);
    assert_eq!(tabular.number_of_columns(), 2);
    assert_eq!(tabular.position(), 7);
    assert!(tabular.is_empty());
    assert!(tabular.column(0).expect("a column").is_empty());
}

#[test]
fn sk_sliced_record_batch_imports_the_logical_window() {
    let sliced = sample_batch().slice(1, 2);
    assert_eq!(sliced.num_rows(), 2);

    let tabular =
        sk_tabular_batch_from_record_batch(&sliced, SKMissingValuePolicy::Reject, 3, false)
            .expect("a sliced import");

    assert_eq!(tabular.number_of_rows(), 2);
    assert_eq!(
        tabular.column_named("double"),
        Some(&SKColumnData::Float64(vec![Some(20.0), Some(30.0)]))
    );
    assert_eq!(
        tabular.column_named("species"),
        Some(&SKColumnData::Nominal(vec![
            Some("b".to_string()),
            Some("c".to_string())
        ]))
    );
    assert_eq!(tabular.position(), 3);
    assert!(!tabular.is_final());
}
