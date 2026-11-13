//! Acceptance scenarios for conversion: tiny and large tables, both scalars,
//! nullable/categorical columns and concurrency.

use std::sync::Arc;
use std::thread;

use ndarray::array;
use sciencekit_common::{SKLazySource, SKSourceProvider};

use super::fixtures::{numeric_batch, numeric_selection};
use crate::batch::{SKFeatureLayout, SKTabularBatch};
use crate::columns::SKColumnData;
use crate::missing::SKMissingValuePolicy;
use crate::schema::{SKColumnRole, SKColumnSchema, SKLogicalType, SKTabularSchema};
use crate::selection::SKColumnSelection;
use crate::source::{SKInMemoryTabularSource, SKTabularSourceProvider};

/// A tiny table converts correctly for both model scalars.
#[test]
fn tiny_table_converts_for_both_scalars() {
    let batch = numeric_batch(3);
    assert_eq!(
        batch
            .to_dense::<f64>(&numeric_selection(), SKMissingValuePolicy::Reject)
            .unwrap(),
        array![
            [0.0_f64, 0.0, 0.0, 1.0],
            [0.5, 1.0, 1.0, 0.0],
            [1.0, 2.0, 2.0, 1.0]
        ]
    );
    let single = batch
        .to_dense::<f32>(&numeric_selection(), SKMissingValuePolicy::Reject)
        .unwrap();
    assert_eq!(single[[2, 1]], 2.0_f32);
    assert_eq!(single[[2, 2]], 2.0_f32);
}

/// A large table converts without losing values, and its stream preserves
/// order across many blocks.
#[test]
fn large_table_converts_and_streams_in_order() {
    let rows = 20_000usize;
    let matrix = numeric_batch(rows)
        .to_dense::<f64>(&numeric_selection(), SKMissingValuePolicy::Reject)
        .unwrap();
    assert_eq!(matrix.dim(), (rows, 4));
    assert_eq!(matrix[[rows - 1, 1]], (rows - 1) as f64);
    assert_eq!(matrix[[rows - 1, 2]], (rows - 1) as f64);

    let block_rows = 2_000usize;
    let blocks: Vec<SKTabularBatch> = (0..rows / block_rows)
        .map(|block| {
            let start = block * block_rows;
            let schema = SKTabularSchema::new(vec![SKColumnSchema::new(
                "double",
                SKLogicalType::Float64,
                SKColumnRole::Feature,
            )])
            .unwrap();
            let doubles: Vec<Option<f64>> = (start..start + block_rows)
                .map(|row| Some(row as f64))
                .collect();
            SKTabularBatch::new(
                schema,
                vec![SKColumnData::Float64(doubles)],
                block,
                block + 1 == rows / block_rows,
            )
            .unwrap()
        })
        .collect();

    let mut provider = SKTabularSourceProvider::new(
        SKInMemoryTabularSource::new(blocks),
        SKColumnSelection::builder()
            .feature("double")
            .build()
            .unwrap(),
        SKMissingValuePolicy::Reject,
        SKFeatureLayout::Dense,
    );
    let mut source = SKSourceProvider::<f64>::open(&mut provider).unwrap();
    let mut positions = Vec::new();
    let mut total_rows = 0usize;
    while let Some(block) = source.next_batch().unwrap() {
        positions.push(block.position());
        total_rows += block.number_of_rows();
    }
    assert_eq!(positions, (0..10).collect::<Vec<_>>());
    assert_eq!(total_rows, rows);
}

/// Nullable numeric and categorical columns follow one declared policy.
#[test]
fn nullable_and_categorical_columns_follow_the_policy() {
    let schema = SKTabularSchema::new(vec![
        SKColumnSchema::new("price", SKLogicalType::Float64, SKColumnRole::Feature),
        SKColumnSchema::new("species", SKLogicalType::Nominal, SKColumnRole::Feature),
    ])
    .unwrap();
    let batch = SKTabularBatch::new(
        schema,
        vec![
            SKColumnData::Float64(vec![Some(1.0), None]),
            SKColumnData::Nominal(vec![Some("cat".into()), None]),
        ],
        0,
        true,
    )
    .unwrap();

    let dense = batch
        .to_dense::<f64>(
            &SKColumnSelection::builder()
                .feature("price")
                .build()
                .unwrap(),
            SKMissingValuePolicy::NumericNaN,
        )
        .unwrap();
    assert!(dense[[1, 0]].is_nan());

    let categorical = batch
        .extract_categorical(
            &SKColumnSelection::builder()
                .feature("species")
                .build()
                .unwrap(),
            SKMissingValuePolicy::CategoricalMissingCode,
        )
        .unwrap();
    assert_eq!(
        categorical
            .codes()
            .unwrap()
            .iter()
            .copied()
            .collect::<Vec<_>>(),
        vec![0, -1]
    );
}

/// Concurrent conversions of the same batch produce identical blocks.
#[test]
fn concurrent_conversions_agree() {
    let batch = Arc::new(numeric_batch(1_000));
    let expected = Arc::new(
        batch
            .to_dense::<f64>(&numeric_selection(), SKMissingValuePolicy::Reject)
            .unwrap(),
    );

    let mut handles = Vec::new();
    for _worker in 0..4 {
        let batch = Arc::clone(&batch);
        let expected = Arc::clone(&expected);
        handles.push(thread::spawn(move || {
            let converted = batch
                .to_dense::<f64>(&numeric_selection(), SKMissingValuePolicy::Reject)
                .unwrap();
            assert_eq!(converted, *expected);
        }));
    }
    for handle in handles {
        handle.join().unwrap();
    }
}
