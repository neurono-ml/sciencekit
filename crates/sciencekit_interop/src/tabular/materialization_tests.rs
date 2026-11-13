//! Tests for batch-to-dataset materialization (spec `tabular-sources`).

use ndarray::array;
use sciencekit_common::{SKError, SKFeatureBlock};

use super::{SKColumnKind, SKTabularSourceConfig};
use crate::batch::{SKFeatureLayout, SKTabularBatch};
use crate::columns::SKColumnData;
use crate::schema::{SKColumnRole, SKColumnSchema, SKLogicalType, SKTabularSchema};

fn mixed_batch() -> SKTabularBatch {
    let schema = SKTabularSchema::new(vec![
        SKColumnSchema::new("x", SKLogicalType::Float64, SKColumnRole::Feature),
        SKColumnSchema::new("count", SKLogicalType::Integer64, SKColumnRole::Feature),
        SKColumnSchema::new("flag", SKLogicalType::Boolean, SKColumnRole::Feature),
        SKColumnSchema::new("label", SKLogicalType::Nominal, SKColumnRole::Feature),
        SKColumnSchema::new("note", SKLogicalType::Float64, SKColumnRole::Ignored),
    ])
    .unwrap();
    SKTabularBatch::new(
        schema,
        vec![
            SKColumnData::Float64(vec![Some(1.0), Some(2.0), Some(3.0)]),
            SKColumnData::Integer64(vec![Some(10), Some(20), Some(30)]),
            SKColumnData::Boolean(vec![Some(true), Some(false), Some(true)]),
            SKColumnData::Nominal(vec![
                Some("a".to_string()),
                Some("b".to_string()),
                Some("a".to_string()),
            ]),
            SKColumnData::Float64(vec![Some(9.0), Some(9.0), Some(9.0)]),
        ],
        0,
        true,
    )
    .unwrap()
}

fn config() -> SKTabularSourceConfig {
    SKTabularSourceConfig::builder()
        .column("label", SKColumnKind::Categorical)
        .column("x", SKColumnKind::Numeric)
        .column("count", SKColumnKind::Numeric)
        .column("flag", SKColumnKind::Boolean)
        .column("note", SKColumnKind::Ignored)
        .build()
        .unwrap()
}

/// Numeric and boolean columns form a dense block in declaration order.
#[test]
fn tabular_source_numeric_columns_form_feature_block_in_order() {
    let dataset = mixed_batch().to_tabular_dataset::<f64>(&config()).unwrap();
    match dataset.features() {
        SKFeatureBlock::Dense(matrix) => {
            assert_eq!(
                matrix,
                array![[1.0, 10.0, 1.0], [2.0, 20.0, 0.0], [3.0, 30.0, 1.0]]
            );
        }
        other => panic!("expected a dense block, found {other:?}"),
    }
    assert_eq!(dataset.schema().len(), 4);
    assert_eq!(dataset.number_of_rows(), 3);
}

/// Ignored columns never reach the outputs, including the projected schema.
#[test]
fn tabular_source_ignored_columns_are_excluded() {
    let dataset = mixed_batch().to_tabular_dataset::<f64>(&config()).unwrap();
    assert!(dataset.schema().column("note").is_none());
    assert_eq!(dataset.number_of_categorical_columns(), 1);
}

/// Categorical columns form an owned nominal block with a label table.
#[test]
fn tabular_source_categorical_columns_form_owned_block() {
    let dataset = mixed_batch().to_tabular_dataset::<f64>(&config()).unwrap();
    let categorical = dataset.categorical().unwrap();
    assert_eq!(categorical.shape(), (3, 1));
    assert_eq!(categorical.codes().column(0).to_vec(), vec![0, 1, 0]);
    assert_eq!(categorical.label_table(0).unwrap().index_of("b"), Some(1));
}

/// The sparse layout reaches the dataset unchanged and agrees with dense.
#[test]
fn tabular_source_honors_sparse_layout() {
    let sparse_config = SKTabularSourceConfig::builder()
        .column("x", SKColumnKind::Numeric)
        .column("count", SKColumnKind::Numeric)
        .column("flag", SKColumnKind::Boolean)
        .layout(SKFeatureLayout::Sparse)
        .build()
        .unwrap();
    let sparse = mixed_batch()
        .to_tabular_dataset::<f64>(&sparse_config)
        .unwrap();
    assert!(sparse.features().is_sparse());

    let dense_config = SKTabularSourceConfig::builder()
        .column("x", SKColumnKind::Numeric)
        .column("count", SKColumnKind::Numeric)
        .column("flag", SKColumnKind::Boolean)
        .build()
        .unwrap();
    let dense = mixed_batch()
        .to_tabular_dataset::<f64>(&dense_config)
        .unwrap();
    let dense_matrix = match dense.features() {
        SKFeatureBlock::Dense(matrix) => matrix.clone(),
        other => panic!("expected a dense block, found {other:?}"),
    };

    match sparse.features() {
        SKFeatureBlock::Sparse(matrix) => {
            assert_eq!((matrix.rows(), matrix.cols()), (3, 3));
            // The false boolean is a real zero and stays unstored, exactly as
            // in CSR: eight of nine entries are non-zero.
            assert_eq!(matrix.nnz(), 8);
            assert_eq!(matrix.get(1, 2), None);
            for row in 0..3 {
                for column in 0..3 {
                    let value = matrix.get(row, column).copied().unwrap_or(0.0);
                    assert_eq!(value, dense_matrix[[row, column]]);
                }
            }
        }
        other => panic!("expected a sparse block, found {other:?}"),
    }
}

/// A configured name absent from the source is reported before conversion.
#[test]
fn tabular_source_unknown_column_is_rejected() {
    let config = SKTabularSourceConfig::builder()
        .column("absent", SKColumnKind::Numeric)
        .build()
        .unwrap();
    let error = mixed_batch()
        .to_tabular_dataset::<f64>(&config)
        .unwrap_err();
    match error {
        SKError::UnknownColumn { column, suggestion } => {
            assert_eq!(column, "absent");
            assert!(suggestion.contains("x"));
        }
        other => panic!("expected UnknownColumn, found {other:?}"),
    }
}

/// Declaring a nominal column numeric is a type mismatch, not a panic.
#[test]
fn tabular_source_kind_mismatch_names_the_column() {
    let config = SKTabularSourceConfig::builder()
        .column("label", SKColumnKind::Numeric)
        .build()
        .unwrap();
    let error = mixed_batch()
        .to_tabular_dataset::<f64>(&config)
        .unwrap_err();
    match error {
        SKError::UnsupportedColumnType {
            column,
            logical_type,
            ..
        } => {
            assert_eq!(column, "label");
            assert_eq!(logical_type, "nominal");
        }
        other => panic!("expected UnsupportedColumnType, found {other:?}"),
    }
}

/// A categorical-only configuration still produces a zero-width dense block.
#[test]
fn tabular_source_categorical_only_has_zero_numeric_columns() {
    let config = SKTabularSourceConfig::builder()
        .column("label", SKColumnKind::Categorical)
        .build()
        .unwrap();
    let dataset = mixed_batch().to_tabular_dataset::<f64>(&config).unwrap();
    assert_eq!(dataset.number_of_feature_columns(), 0);
    assert_eq!(dataset.number_of_rows(), 3);
}
