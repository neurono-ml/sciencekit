//! Tests for target extraction through the library's target view (spec
//! `dataset-interop-core`).

use sciencekit_common::{SKError, SKTargetView};

use super::SKTabularBatch;
use crate::columns::SKColumnData;
use crate::missing::SKMissingValuePolicy;
use crate::schema::{SKColumnRole, SKColumnSchema, SKLogicalType, SKTabularSchema};
use crate::selection::SKColumnSelection;

fn batch(target_type: SKLogicalType, target: SKColumnData) -> SKTabularBatch {
    let schema = SKTabularSchema::new(vec![
        SKColumnSchema::new("x", SKLogicalType::Float64, SKColumnRole::Feature),
        SKColumnSchema::new("y", target_type, SKColumnRole::Target),
    ])
    .unwrap();
    SKTabularBatch::new(
        schema,
        vec![SKColumnData::Float64(vec![Some(1.0); target.len()]), target],
        0,
        true,
    )
    .unwrap()
}

fn supervised_selection() -> SKColumnSelection {
    SKColumnSelection::builder()
        .feature("x")
        .target("y")
        .build()
        .unwrap()
}

/// Continuous targets surface as the continuous view.
#[test]
fn continuous_target_extracts_to_the_continuous_view() {
    let batch = batch(
        SKLogicalType::Float64,
        SKColumnData::Float64(vec![Some(0.5), Some(1.5)]),
    );
    let extraction = batch
        .extract_target::<f64>(&supervised_selection(), SKMissingValuePolicy::Reject)
        .unwrap();
    assert_eq!(extraction.len(), 2);
    extraction.with_view(|view| match view {
        SKTargetView::Continuous(values) => {
            assert_eq!(values.to_vec(), vec![0.5, 1.5]);
        }
        other => panic!("expected a continuous view, got {other:?}"),
    });
}

/// Integer targets surface as the integer view.
#[test]
fn integer_target_extracts_to_the_integer_view() {
    let batch = batch(
        SKLogicalType::Integer64,
        SKColumnData::Integer64(vec![Some(0), Some(2)]),
    );
    let extraction = batch
        .extract_target::<f64>(&supervised_selection(), SKMissingValuePolicy::Reject)
        .unwrap();
    extraction.with_view(|view| match view {
        SKTargetView::Integer(values) => assert_eq!(values.to_vec(), vec![0, 2]),
        other => panic!("expected an integer view, got {other:?}"),
    });
}

/// Nominal targets canonicalize deterministically and round-trip through the
/// label table.
#[test]
fn nominal_target_is_canonicalized_deterministically() {
    let labels = vec![
        Some("cat".to_string()),
        Some("dog".to_string()),
        Some("cat".to_string()),
        Some("bird".to_string()),
    ];
    let batch = batch(SKLogicalType::Nominal, SKColumnData::Nominal(labels));
    let selection = supervised_selection();
    let first = batch
        .extract_target::<f64>(&selection, SKMissingValuePolicy::Reject)
        .unwrap();
    let second = batch
        .extract_target::<f64>(&selection, SKMissingValuePolicy::Reject)
        .unwrap();

    assert_eq!(first, second, "the same input must produce the same table");
    assert_eq!(
        first.label_table().unwrap(),
        second.label_table().unwrap()
    );
    assert_eq!(
        first.canonical_indices().unwrap().to_vec(),
        vec![0, 1, 0, 2]
    );
    let table = first.label_table().unwrap();
    assert_eq!(table.labels().collect::<Vec<_>>(), vec!["cat", "dog", "bird"]);
    first.with_view(|view| match view {
        SKTargetView::Nominal(text) => assert_eq!(text, &["cat", "dog", "cat", "bird"]),
        other => panic!("expected a nominal view, got {other:?}"),
    });
    assert_eq!(
        first.to_scalar_values().unwrap().to_vec(),
        vec![0.0, 1.0, 0.0, 2.0]
    );
}

/// A nominal null under the categorical policy keeps the missing sentinel.
#[test]
fn missing_nominal_target_uses_the_missing_code() {
    let batch = batch(
        SKLogicalType::Nominal,
        SKColumnData::Nominal(vec![Some("cat".to_string()), None]),
    );
    let extraction = batch
        .extract_target::<f64>(
            &supervised_selection(),
            SKMissingValuePolicy::CategoricalMissingCode,
        )
        .unwrap();
    assert_eq!(
        extraction.canonical_indices().unwrap().to_vec(),
        vec![0, -1]
    );
    extraction.with_view(|view| match view {
        SKTargetView::Integer(values) => assert_eq!(values.to_vec(), vec![0, -1]),
        other => panic!("expected the integer fallback, got {other:?}"),
    });
}

/// A nominal null under reject fails and names the column.
#[test]
fn missing_nominal_target_under_reject_fails() {
    let batch = batch(
        SKLogicalType::Nominal,
        SKColumnData::Nominal(vec![Some("cat".to_string()), None]),
    );
    let error = batch
        .extract_target::<f64>(&supervised_selection(), SKMissingValuePolicy::Reject)
        .unwrap_err();
    assert!(matches!(error, SKError::NullValueInColumn { .. }));
    assert!(error.to_string().contains('y'));
}

/// Category-coded targets keep their codes and accept the sentinel.
#[test]
fn category_code_target_keeps_codes() {
    let batch = batch(
        SKLogicalType::CategoryCode,
        SKColumnData::CategoryCode(vec![Some(0), None, Some(2)]),
    );
    let extraction = batch
        .extract_target::<f64>(
            &supervised_selection(),
            SKMissingValuePolicy::CategoricalMissingCode,
        )
        .unwrap();
    assert_eq!(extraction.canonical_indices(), None);
    extraction.with_view(|view| match view {
        SKTargetView::Integer(values) => assert_eq!(values.to_vec(), vec![0, -1, 2]),
        other => panic!("expected an integer view, got {other:?}"),
    });
}

/// An unsupervised selection has no target to extract.
#[test]
fn missing_target_selection_is_rejected() {
    let selection = SKColumnSelection::builder().feature("x").build().unwrap();
    let batch = batch(
        SKLogicalType::Float64,
        SKColumnData::Float64(vec![Some(1.0)]),
    );
    let error = batch
        .extract_target::<f64>(&selection, SKMissingValuePolicy::Reject)
        .unwrap_err();
    assert!(matches!(error, SKError::UnsupportedOperation { .. }));
}
