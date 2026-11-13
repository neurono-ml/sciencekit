//! Tests for categorical extraction through the library's categorical view
//! (spec `dataset-interop-core`).

use sciencekit_common::{SK_CATEGORICAL_MISSING_CODE, SKCategoricalView, SKError};

use super::SKTabularBatch;
use crate::columns::SKColumnData;
use crate::missing::SKMissingValuePolicy;
use crate::schema::{SKColumnRole, SKColumnSchema, SKLogicalType, SKTabularSchema};
use crate::selection::SKColumnSelection;

fn batch(columns: Vec<(&str, SKLogicalType, SKColumnData)>) -> SKTabularBatch {
    let schema = SKTabularSchema::new(
        columns
            .iter()
            .map(|(name, logical_type, _)| {
                SKColumnSchema::new(*name, *logical_type, SKColumnRole::Feature)
            })
            .collect(),
    )
    .unwrap();
    let columns = columns.into_iter().map(|(_, _, data)| data).collect();
    SKTabularBatch::new(schema, columns, 0, true).unwrap()
}

fn selection(names: &[&str]) -> SKColumnSelection {
    let mut builder = SKColumnSelection::builder();
    for name in names {
        builder = builder.feature(*name);
    }
    builder.build().unwrap()
}

/// Fully present nominal columns surface through the nominal view.
#[test]
fn nominal_columns_surface_as_a_nominal_view() {
    let batch = batch(vec![
        (
            "species",
            SKLogicalType::Nominal,
            SKColumnData::Nominal(vec![Some("cat".into()), Some("dog".into())]),
        ),
        (
            "color",
            SKLogicalType::Nominal,
            SKColumnData::Nominal(vec![Some("red".into()), Some("red".into())]),
        ),
    ]);
    let extraction = batch
        .extract_categorical(
            &selection(&["species", "color"]),
            SKMissingValuePolicy::Reject,
        )
        .unwrap();
    assert_eq!(extraction.shape(), (2, 2));
    assert!(extraction.codes().is_none());
    extraction.with_view(|view| {
        assert_eq!(view.representation(), "nominal");
        assert_eq!(view.shape(), (2, 2));
        match view {
            SKCategoricalView::Nominal(matrix) => {
                assert_eq!(matrix.value(0, 0), "cat");
                assert_eq!(matrix.value(1, 0), "dog");
                assert_eq!(matrix.value(1, 1), "red");
            }
            other => panic!("expected a nominal view, got {other:?}"),
        }
    });
}

/// Category codes keep their values, including the missing sentinel.
#[test]
fn category_codes_keep_codes_and_sentinel() {
    let batch = batch(vec![(
        "species_code",
        SKLogicalType::CategoryCode,
        SKColumnData::CategoryCode(vec![Some(0), None, Some(SK_CATEGORICAL_MISSING_CODE)]),
    )]);
    let extraction = batch
        .extract_categorical(
            &selection(&["species_code"]),
            SKMissingValuePolicy::CategoricalMissingCode,
        )
        .unwrap();
    assert_eq!(extraction.shape(), (3, 1));
    let codes = extraction.codes().unwrap();
    assert_eq!(codes.iter().copied().collect::<Vec<_>>(), vec![0, -1, -1]);
    extraction.with_view(|view| {
        assert_eq!(view.representation(), "integer");
    });
}

/// A nominal column with a null under the categorical policy becomes codes and
/// carries the sentinel.
#[test]
fn nominal_nulls_become_codes_with_the_sentinel() {
    let batch = batch(vec![(
        "species",
        SKLogicalType::Nominal,
        SKColumnData::Nominal(vec![Some("cat".into()), None, Some("dog".into())]),
    )]);
    let extraction = batch
        .extract_categorical(
            &selection(&["species"]),
            SKMissingValuePolicy::CategoricalMissingCode,
        )
        .unwrap();
    let codes = extraction.codes().unwrap();
    assert_eq!(codes.iter().copied().collect::<Vec<_>>(), vec![0, -1, 1]);
    let tables = extraction.label_tables();
    assert_eq!(tables.len(), 1);
    assert_eq!(
        tables[0].as_ref().unwrap().labels().collect::<Vec<_>>(),
        vec!["cat", "dog"]
    );
}

/// A nominal null under reject fails and names the column.
#[test]
fn nominal_null_under_reject_is_rejected() {
    let batch = batch(vec![(
        "species",
        SKLogicalType::Nominal,
        SKColumnData::Nominal(vec![Some("cat".into()), None]),
    )]);
    let error = batch
        .extract_categorical(&selection(&["species"]), SKMissingValuePolicy::Reject)
        .unwrap_err();
    assert!(matches!(error, SKError::NullValueInColumn { .. }));
    assert!(error.to_string().contains("species"));
}

/// A numeric column cannot enter the categorical view.
#[test]
fn numeric_columns_are_unsupported_on_the_categorical_path() {
    let batch = batch(vec![(
        "x",
        SKLogicalType::Float64,
        SKColumnData::Float64(vec![Some(1.0)]),
    )]);
    let error = batch
        .extract_categorical(&selection(&["x"]), SKMissingValuePolicy::Reject)
        .unwrap_err();
    match &error {
        SKError::UnsupportedColumnType {
            column, suggestion, ..
        } => {
            assert_eq!(column, "x");
            assert!(suggestion.contains("nominal"));
        }
        other => panic!("expected an unsupported-column-type error, got {other:?}"),
    }
}

/// Mixed nominal and coded columns share one integer extraction.
#[test]
fn mixed_categorical_columns_share_the_integer_view() {
    let batch = batch(vec![
        (
            "species",
            SKLogicalType::Nominal,
            SKColumnData::Nominal(vec![Some("a".into()), Some("b".into())]),
        ),
        (
            "region",
            SKLogicalType::CategoryCode,
            SKColumnData::CategoryCode(vec![Some(7), Some(8)]),
        ),
    ]);
    let extraction = batch
        .extract_categorical(
            &selection(&["species", "region"]),
            SKMissingValuePolicy::Reject,
        )
        .unwrap();
    let codes = extraction.codes().unwrap();
    assert_eq!(codes.iter().copied().collect::<Vec<_>>(), vec![0, 7, 1, 8]);
    assert_eq!(extraction.label_tables().len(), 2);
    assert!(extraction.label_tables()[0].is_some());
    assert!(extraction.label_tables()[1].is_none());
}
