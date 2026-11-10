//! Tests for dense materialization (spec `dataset-interop-core`).

use ndarray::array;

use super::SKTabularBatch;
use crate::columns::SKColumnData;
use crate::missing::SKMissingValuePolicy;
use crate::schema::{SKColumnRole, SKColumnSchema, SKLogicalType, SKTabularSchema};
use crate::selection::SKColumnSelection;

fn table_batch() -> SKTabularBatch {
    let schema = SKTabularSchema::new(vec![
        SKColumnSchema::new("a", SKLogicalType::Float64, SKColumnRole::Feature),
        SKColumnSchema::new("b", SKLogicalType::Integer64, SKColumnRole::Feature),
        SKColumnSchema::new("ignored", SKLogicalType::Float64, SKColumnRole::Ignored),
    ])
    .unwrap();
    SKTabularBatch::new(
        schema,
        vec![
            SKColumnData::Float64(vec![Some(1.0), Some(2.0)]),
            SKColumnData::Integer64(vec![Some(10), Some(20)]),
            SKColumnData::Float64(vec![Some(99.0), Some(99.0)]),
        ],
        0,
        true,
    )
    .unwrap()
}

/// Dense materialization puts selected features in selection order.
#[test]
fn dense_materialization_preserves_selection_order() {
    let selection = SKColumnSelection::builder()
        .feature("b")
        .feature("a")
        .build()
        .unwrap();
    let matrix = table_batch()
        .to_dense::<f64>(&selection, SKMissingValuePolicy::Reject)
        .unwrap();
    assert_eq!(matrix, array![[10_f64, 1.0], [20.0, 2.0]]);
}

/// The same conversion also runs for `f32`, including integer coercion.
#[test]
fn dense_materialization_runs_for_f32() {
    let selection = SKColumnSelection::builder()
        .feature("a")
        .feature("b")
        .build()
        .unwrap();
    let matrix = table_batch()
        .to_dense::<f32>(&selection, SKMissingValuePolicy::Reject)
        .unwrap();
    assert_eq!(matrix, array![[1.0_f32, 10.0], [2.0, 20.0]]);
}

/// The materialized buffer is independent of the batch: dropping the source
/// leaves the result valid.
#[test]
fn dense_materialization_does_not_alias_the_batch() {
    let selection = SKColumnSelection::builder().feature("a").build().unwrap();
    let matrix = {
        let batch = table_batch();
        batch
            .to_dense::<f64>(&selection, SKMissingValuePolicy::Reject)
            .unwrap()
    };
    assert_eq!(matrix, array![[1.0_f64], [2.0]]);
}

/// An empty batch materializes an empty matrix with the full feature width.
#[test]
fn empty_batch_materializes_an_empty_matrix() {
    let schema = SKTabularSchema::new(vec![SKColumnSchema::new(
        "x",
        SKLogicalType::Float64,
        SKColumnRole::Feature,
    )])
    .unwrap();
    let batch =
        SKTabularBatch::new(schema, vec![SKColumnData::Float64(Vec::new())], 0, true).unwrap();
    let selection = SKColumnSelection::builder().feature("x").build().unwrap();
    let matrix = batch
        .to_dense::<f64>(&selection, SKMissingValuePolicy::Reject)
        .unwrap();
    assert_eq!(matrix.dim(), (0, 1));
}
