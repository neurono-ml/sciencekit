//! Tests for missing-value and unsupported-type mapping in the tabular
//! contract (spec `tabular-sources`).

use sciencekit_common::{SK_CATEGORICAL_MISSING_CODE, SKError, SKFeatureBlock};

use super::{SKColumnKind, SKTabularSourceConfig};
use crate::batch::SKTabularBatch;
use crate::columns::SKColumnData;
use crate::missing::SKMissingValuePolicy;
use crate::schema::{SKColumnRole, SKColumnSchema, SKLogicalType, SKTabularSchema};

fn nullable_batch() -> SKTabularBatch {
    let schema = SKTabularSchema::new(vec![
        SKColumnSchema::new("x", SKLogicalType::Float64, SKColumnRole::Feature),
        SKColumnSchema::new("label", SKLogicalType::Nominal, SKColumnRole::Feature),
    ])
    .unwrap();
    SKTabularBatch::new(
        schema,
        vec![
            SKColumnData::Float64(vec![Some(1.0), None]),
            SKColumnData::Nominal(vec![Some("a".to_string()), None]),
        ],
        0,
        true,
    )
    .unwrap()
}

/// A numeric null becomes NaN under the numeric policy.
#[test]
fn tabular_missing_numeric_null_becomes_nan() {
    let config = SKTabularSourceConfig::builder()
        .column("x", SKColumnKind::Numeric)
        .missing_value_policy(SKMissingValuePolicy::NumericNaN)
        .build()
        .unwrap();
    let dataset = nullable_batch().to_tabular_dataset::<f64>(&config).unwrap();
    match dataset.features() {
        SKFeatureBlock::Dense(matrix) => {
            assert_eq!(matrix[[0, 0]], 1.0);
            assert!(matrix[[1, 0]].is_nan());
        }
        other => panic!("expected a dense block, found {other:?}"),
    }
}

/// The default reject policy stops at the first null and names the column.
#[test]
fn tabular_missing_reject_policy_names_the_column() {
    let config = SKTabularSourceConfig::builder()
        .column("x", SKColumnKind::Numeric)
        .build()
        .unwrap();
    let error = nullable_batch()
        .to_tabular_dataset::<f64>(&config)
        .unwrap_err();
    match error {
        SKError::NullValueInColumn { column, policy } => {
            assert_eq!(column, "x");
            assert_eq!(policy, "reject");
        }
        other => panic!("expected NullValueInColumn, found {other:?}"),
    }
}

/// A categorical null becomes the documented missing sentinel.
#[test]
fn tabular_missing_categorical_null_becomes_sentinel() {
    let config = SKTabularSourceConfig::builder()
        .column("label", SKColumnKind::Categorical)
        .missing_value_policy(SKMissingValuePolicy::CategoricalMissingCode)
        .build()
        .unwrap();
    let dataset = nullable_batch().to_tabular_dataset::<f64>(&config).unwrap();
    let categorical = dataset.categorical().unwrap();
    assert_eq!(categorical.codes()[[0, 0]], 0);
    assert_eq!(categorical.codes()[[1, 0]], SK_CATEGORICAL_MISSING_CODE);
}

/// A boolean column with a null follows the numeric policy.
#[test]
fn tabular_missing_boolean_null_becomes_nan() {
    let schema = SKTabularSchema::new(vec![SKColumnSchema::new(
        "flag",
        SKLogicalType::Boolean,
        SKColumnRole::Feature,
    )])
    .unwrap();
    let batch = SKTabularBatch::new(
        schema,
        vec![SKColumnData::Boolean(vec![Some(true), None])],
        0,
        true,
    )
    .unwrap();
    let config = SKTabularSourceConfig::builder()
        .column("flag", SKColumnKind::Boolean)
        .missing_value_policy(SKMissingValuePolicy::NumericNaN)
        .build()
        .unwrap();
    let dataset = batch.to_tabular_dataset::<f32>(&config).unwrap();
    match dataset.features() {
        SKFeatureBlock::Dense(matrix) => {
            assert_eq!(matrix[[0, 0]], 1.0);
            assert!(matrix[[1, 0]].is_nan());
        }
        other => panic!("expected a dense block, found {other:?}"),
    }
}

/// An empty source is rejected instead of producing a zero-row dataset.
#[test]
fn tabular_source_rejects_empty_input() {
    let batch = SKTabularBatch::new(
        crate::schema::SKTabularSchema::new(vec![SKColumnSchema::new(
            "x",
            SKLogicalType::Float64,
            SKColumnRole::Feature,
        )])
        .unwrap(),
        vec![SKColumnData::Float64(Vec::new())],
        0,
        true,
    )
    .unwrap();
    let config = SKTabularSourceConfig::builder()
        .column("x", SKColumnKind::Numeric)
        .build()
        .unwrap();
    let error = batch.to_tabular_dataset::<f64>(&config).unwrap_err();
    assert!(matches!(error, SKError::EmptyInput { .. }));
}

/// A non-finite value that is not the missing sentinel is rejected.
#[test]
fn tabular_source_rejects_infinite_values() {
    let batch = SKTabularBatch::new(
        crate::schema::SKTabularSchema::new(vec![SKColumnSchema::new(
            "x",
            SKLogicalType::Float64,
            SKColumnRole::Feature,
        )])
        .unwrap(),
        vec![SKColumnData::Float64(vec![Some(f64::INFINITY)])],
        0,
        true,
    )
    .unwrap();
    let config = SKTabularSourceConfig::builder()
        .column("x", SKColumnKind::Numeric)
        .build()
        .unwrap();
    let error = batch.to_tabular_dataset::<f64>(&config).unwrap_err();
    assert!(matches!(error, SKError::NonFiniteValue { .. }));
}
