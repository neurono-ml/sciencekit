//! Tests for conversion into owned streaming blocks (spec
//! `dataset-interop-core`).

use ndarray::array;

use super::{SKFeatureLayout, SKTabularBatch};
use crate::columns::SKColumnData;
use crate::missing::SKMissingValuePolicy;
use crate::schema::{SKColumnRole, SKColumnSchema, SKLogicalType, SKTabularSchema};
use crate::selection::SKColumnSelection;

fn batch() -> SKTabularBatch {
    let schema = SKTabularSchema::new(vec![
        SKColumnSchema::new("a", SKLogicalType::Float64, SKColumnRole::Feature),
        SKColumnSchema::new("b", SKLogicalType::Float64, SKColumnRole::Feature),
        SKColumnSchema::new("label", SKLogicalType::Nominal, SKColumnRole::Target),
    ])
    .unwrap();
    SKTabularBatch::new(
        schema,
        vec![
            SKColumnData::Float64(vec![Some(1.0), Some(2.0)]),
            SKColumnData::Float64(vec![Some(0.0), Some(0.0)]),
            SKColumnData::Nominal(vec![Some("cat".to_string()), Some("dog".to_string())]),
        ],
        4,
        false,
    )
    .unwrap()
}

fn selection() -> SKColumnSelection {
    SKColumnSelection::builder()
        .feature("a")
        .feature("b")
        .target("label")
        .build()
        .unwrap()
}

/// The data batch owns its values and preserves position and final metadata.
#[test]
fn data_batch_owns_values_and_metadata() {
    let selection = SKColumnSelection::builder()
        .feature("a")
        .feature("b")
        .build()
        .unwrap();
    let data_batch = {
        let batch = batch();
        batch
            .to_data_batch::<f64>(
                &selection,
                SKMissingValuePolicy::Reject,
                SKFeatureLayout::Dense,
            )
            .unwrap()
    };
    assert_eq!(data_batch.position(), 4);
    assert!(!data_batch.is_final());
    assert_eq!(data_batch.number_of_rows(), 2);
    assert_eq!(data_batch.number_of_columns(), 2);
    match data_batch.view() {
        sciencekit_common::SKDataView::Dense(matrix) => {
            assert_eq!(matrix, array![[1.0_f64, 0.0], [2.0, 0.0]].view());
        }
        other => panic!("expected a dense view, got {other:?}"),
    }
}

/// A sparse conversion produces a CSR streaming block.
#[test]
fn data_batch_can_be_sparse() {
    let selection = SKColumnSelection::builder()
        .feature("a")
        .feature("b")
        .build()
        .unwrap();
    let data_batch = batch()
        .to_data_batch::<f64>(
            &selection,
            SKMissingValuePolicy::Reject,
            SKFeatureLayout::Sparse,
        )
        .unwrap();
    assert!(data_batch.is_sparse());
    assert_eq!(data_batch.number_of_rows(), 2);
    // Only the two non-zero entries of column `a` are stored.
    assert!(data_batch.byte_size() >= 2 * std::mem::size_of::<f64>());
}

/// The supervised batch pairs owned features with canonicalized targets.
#[test]
fn supervised_batch_preserves_features_and_targets() {
    let supervised = {
        let batch = batch();
        batch
            .to_supervised_batch::<f64>(
                &selection(),
                SKMissingValuePolicy::Reject,
                SKFeatureLayout::Dense,
            )
            .unwrap()
    };
    assert_eq!(supervised.position(), 4);
    assert!(!supervised.is_final());
    assert_eq!(supervised.targets().to_vec(), vec![0.0, 1.0]);
    assert_eq!(supervised.number_of_columns(), 2);
}

/// A sparse supervised conversion keeps the CSR representation.
#[test]
fn supervised_batch_can_be_sparse() {
    let supervised = batch()
        .to_supervised_batch::<f64>(
            &selection(),
            SKMissingValuePolicy::Reject,
            SKFeatureLayout::Sparse,
        )
        .unwrap();
    assert!(supervised.is_sparse());
    assert_eq!(supervised.targets().len(), 2);
}

/// A supervised conversion without a target is rejected.
#[test]
fn supervised_conversion_requires_a_target() {
    let selection = SKColumnSelection::builder().feature("a").build().unwrap();
    let error = batch()
        .to_supervised_batch::<f64>(
            &selection,
            SKMissingValuePolicy::Reject,
            SKFeatureLayout::Dense,
        )
        .unwrap_err();
    assert!(matches!(
        error,
        sciencekit_common::SKError::UnsupportedOperation { .. }
    ));
}

/// A nominal target becomes canonical indices. The per-batch label table is
/// **not** carried by the supervised batch (documented limitation); callers that
/// need the mapping use `extract_target` directly.
#[test]
fn supervised_conversion_encodes_a_nominal_target_as_indices() {
    let selection = selection();
    let supervised = batch()
        .to_supervised_batch::<f64>(
            &selection,
            SKMissingValuePolicy::CategoricalMissingCode,
            SKFeatureLayout::Dense,
        )
        .unwrap();
    let targets = supervised.targets();
    assert_eq!(targets.len(), 2);
    assert!(targets.iter().all(|value| value.is_finite()));
}
