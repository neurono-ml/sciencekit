//! Tests for the materialized tabular dataset value (spec `tabular-sources`).

use ndarray::{Array2, array};
use sciencekit_common::{
    SKDataView, SKError, SKFeatureBlock, SKLabelTable, sk_canonicalize_labels,
};

use super::{SKNominalMatrixOwned, SKTabularDataset};
use crate::schema::{SKColumnRole, SKColumnSchema, SKLogicalType, SKTabularSchema};

fn label_table() -> SKLabelTable {
    sk_canonicalize_labels(&["a"]).1
}

fn schema() -> SKTabularSchema {
    SKTabularSchema::new(vec![
        SKColumnSchema::new("x", SKLogicalType::Float64, SKColumnRole::Feature),
        SKColumnSchema::new("label", SKLogicalType::Nominal, SKColumnRole::Feature),
    ])
    .unwrap()
}

/// A dataset exposes its projected schema, feature view and categorical block.
#[test]
fn dataset_exposes_schema_features_and_categorical_block() {
    let features = SKFeatureBlock::Dense(array![[1.0_f64], [2.0]]);
    let categorical =
        SKNominalMatrixOwned::new(array![[0_i64], [-1]], vec![Some(label_table())]).unwrap();
    let dataset = SKTabularDataset::new(schema(), features, Some(categorical)).unwrap();

    assert_eq!(dataset.schema().len(), 2);
    assert_eq!(dataset.number_of_rows(), 2);
    assert_eq!(dataset.number_of_feature_columns(), 1);
    assert_eq!(dataset.number_of_categorical_columns(), 1);
    match dataset.feature_view() {
        SKDataView::Dense(view) => assert_eq!(view[[1, 0]], 2.0),
        other => panic!("expected a dense view, found {other:?}"),
    }
    assert_eq!(dataset.categorical().unwrap().codes()[[1, 0]], -1);
}

/// A categorical block with a different row count is rejected.
#[test]
fn dataset_rejects_row_count_mismatch() {
    let features = SKFeatureBlock::Dense(array![[1.0_f64], [2.0]]);
    let categorical =
        SKNominalMatrixOwned::new(array![[0_i64]], vec![Some(label_table())]).unwrap();
    let error = SKTabularDataset::new(schema(), features, Some(categorical)).unwrap_err();
    match error {
        SKError::UnevenColumnLengths {
            column,
            expected,
            found,
        } => {
            assert_eq!(column, "categorical");
            assert_eq!((expected, found), (2, 1));
        }
        other => panic!("expected UnevenColumnLengths, found {other:?}"),
    }
}

/// The schema width must equal the numeric plus categorical width.
#[test]
fn dataset_rejects_schema_width_mismatch() {
    let features = SKFeatureBlock::Dense(array![[1.0_f64]]);
    let error = SKTabularDataset::new(schema(), features, None).unwrap_err();
    assert!(matches!(error, SKError::SchemaMismatch { .. }));
}

/// A dataset without categorical columns reports zero categorical width.
#[test]
fn dataset_without_categorical_columns() {
    let one_column = SKTabularSchema::new(vec![SKColumnSchema::new(
        "x",
        SKLogicalType::Float64,
        SKColumnRole::Feature,
    )])
    .unwrap();
    let features = SKFeatureBlock::Dense(Array2::<f64>::zeros((0, 1)));
    let dataset = SKTabularDataset::new(one_column, features, None).unwrap();
    assert_eq!(dataset.number_of_categorical_columns(), 0);
    assert!(dataset.categorical().is_none());
    assert_eq!(dataset.number_of_rows(), 0);
}
