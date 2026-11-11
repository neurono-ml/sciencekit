//! TDD suite for chunked-column (table-like) import (task 1.4).
//!
//! Arrow's Rust implementation represents a chunked column as a
//! `Vec<ArrayRef>` (the `ChunkedArray` and `Table` types were removed
//! upstream), so these tests exercise that representation directly.

use std::sync::Arc;

use arrow_array::{ArrayRef, Float64Array, Int64Array, StringArray};
use arrow_schema::{DataType, Field, Schema};

use super::import::sk_tabular_batch_from_column_chunks;
use crate::columns::SKColumnData;
use crate::missing::SKMissingValuePolicy;

fn two_column_schema() -> Schema {
    Schema::new(vec![
        Field::new("double", DataType::Float64, false),
        Field::new("species", DataType::Utf8, false),
    ])
}

fn chunked_columns() -> Vec<Vec<ArrayRef>> {
    vec![
        vec![
            Arc::new(Float64Array::from(vec![1.0, 2.0])) as ArrayRef,
            Arc::new(Float64Array::from(vec![3.0])) as ArrayRef,
            Arc::new(Float64Array::from(vec![4.0, 5.0])) as ArrayRef,
        ],
        vec![
            Arc::new(StringArray::from(vec!["a", "b"])) as ArrayRef,
            Arc::new(StringArray::from(vec!["c"])) as ArrayRef,
            Arc::new(StringArray::from(vec!["d", "e"])) as ArrayRef,
        ],
    ]
}

#[test]
fn sk_chunks_concatenate_in_order_with_the_total_length() {
    let tabular = sk_tabular_batch_from_column_chunks(
        &two_column_schema(),
        &chunked_columns(),
        SKMissingValuePolicy::Reject,
        0,
        true,
    )
    .expect("a chunked import");

    assert_eq!(tabular.number_of_rows(), 5);
    assert_eq!(
        tabular.column_named("double"),
        Some(&SKColumnData::Float64(vec![
            Some(1.0),
            Some(2.0),
            Some(3.0),
            Some(4.0),
            Some(5.0)
        ]))
    );
    assert_eq!(
        tabular.column_named("species"),
        Some(&SKColumnData::Nominal(vec![
            Some("a".to_string()),
            Some("b".to_string()),
            Some("c".to_string()),
            Some("d".to_string()),
            Some("e".to_string())
        ]))
    );
}

#[test]
fn sk_zero_chunks_become_an_empty_column() {
    let chunks: Vec<Vec<ArrayRef>> = vec![Vec::new(), Vec::new()];

    let tabular = sk_tabular_batch_from_column_chunks(
        &two_column_schema(),
        &chunks,
        SKMissingValuePolicy::Reject,
        2,
        true,
    )
    .expect("an empty chunked import");

    assert_eq!(tabular.number_of_rows(), 0);
    assert!(tabular.column_named("double").expect("a column").is_empty());
    assert!(
        tabular
            .column_named("species")
            .expect("a column")
            .is_empty()
    );
    assert_eq!(tabular.position(), 2);
}

#[test]
fn sk_single_chunk_is_imported_verbatim() {
    let chunks: Vec<Vec<ArrayRef>> = vec![
        vec![Arc::new(Float64Array::from(vec![7.5])) as ArrayRef],
        vec![Arc::new(StringArray::from(vec!["z"])) as ArrayRef],
    ];

    let tabular = sk_tabular_batch_from_column_chunks(
        &two_column_schema(),
        &chunks,
        SKMissingValuePolicy::Reject,
        0,
        false,
    )
    .expect("a single-chunk import");

    assert_eq!(tabular.number_of_rows(), 1);
    assert_eq!(
        tabular.column_named("double"),
        Some(&SKColumnData::Float64(vec![Some(7.5)]))
    );
}

#[test]
fn sk_mismatched_column_count_or_chunk_type_is_rejected() {
    let wrong_count = vec![vec![Arc::new(Int64Array::from(vec![1])) as ArrayRef]];
    let error = sk_tabular_batch_from_column_chunks(
        &two_column_schema(),
        &wrong_count,
        SKMissingValuePolicy::Reject,
        0,
        false,
    )
    .expect_err("a mismatched column count must be rejected");
    assert!(error.to_string().contains("schema mismatch"), "{error}");

    let wrong_type = vec![
        vec![Arc::new(Int64Array::from(vec![1])) as ArrayRef],
        vec![Arc::new(StringArray::from(vec!["a"])) as ArrayRef],
    ];
    let error = sk_tabular_batch_from_column_chunks(
        &two_column_schema(),
        &wrong_type,
        SKMissingValuePolicy::Reject,
        0,
        false,
    )
    .expect_err("a mismatched chunk type must be rejected");
    assert!(error.to_string().contains("schema mismatch"), "{error}");
}
