//! Tests for the categorical feature boundary (spec `data-view-boundary`).
//!
//! Companion module declared at the end of the categorical `mod.rs`; every test
//! exercises the public surface only.

use ndarray::{Array2, ArrayView2};

use super::{SK_CATEGORICAL_MISSING_CODE, SKCategoricalView, SKNominalMatrixView};
use crate::SKError;

/// A nominal matrix reports its shape and borrows its labels zero-copy.
#[test]
fn nominal_matrix_reports_shape_and_borrows_labels() {
    let row_zero: &[&str] = &["a", "b", "c"];
    let row_one: &[&str] = &["d", "e", "f"];
    let rows: &[&[&str]] = &[row_zero, row_one];

    let view = SKNominalMatrixView::new(rows);
    assert_eq!(view.shape(), (2, 3));
    assert_eq!(view.number_of_rows(), 2);
    assert_eq!(view.number_of_columns(), 3);
    assert_eq!(view.value(1, 2), "f");
    assert_eq!(view.row(0), row_zero);
    assert!(!view.is_empty());
}

/// An empty nominal matrix is empty with zero columns.
#[test]
fn empty_nominal_matrix_is_empty() {
    let rows: &[&[&str]] = &[];
    let view = SKNominalMatrixView::new(rows);
    assert_eq!(view.shape(), (0, 0));
    assert!(view.is_empty());
}

/// A nominal slice converts into the categorical view through the seam.
#[test]
fn nominal_slice_converts_into_categorical_view() {
    let row: &[&str] = &["x", "y"];
    let rows: &[&[&str]] = &[row];
    let view: SKCategoricalView<'_> = rows.try_into().unwrap();
    assert_eq!(view.representation(), "nominal");
    assert_eq!(view.shape(), (1, 2));
    match view {
        SKCategoricalView::Nominal(m) => assert_eq!(m.value(0, 1), "y"),
        other => panic!("expected nominal, got {other:?}"),
    }
}

/// Integer-coded data enters through the same boundary and references the data.
#[test]
fn integer_codes_convert_into_categorical_view() {
    let codes = Array2::from_shape_vec((2, 2), vec![0_i64, 1, 2, 3]).unwrap();
    let view: SKCategoricalView<'_> = ArrayView2::from(&codes).into();
    assert_eq!(view.representation(), "integer");
    assert_eq!(view.shape(), (2, 2));
    match view {
        SKCategoricalView::Integer(m) => assert_eq!(m[[1, 1]], 3),
        other => panic!("expected integer, got {other:?}"),
    }
}

/// An owned integer block enters by borrowing, no copy.
#[test]
fn owned_integer_block_enters_by_borrowing() {
    let codes = Array2::from_shape_vec((1, 1), vec![7_i64]).unwrap();
    let view: SKCategoricalView<'_> = (&codes).into();
    assert_eq!(view.shape(), (1, 1));
}

/// A third-party fallible conversion reports a structured error via the seam.
#[test]
fn third_party_fallible_categorical_conversion_reports_structured_error() {
    struct External(u32);
    impl TryFrom<External> for SKCategoricalView<'_> {
        type Error = SKError;
        fn try_from(value: External) -> Result<Self, Self::Error> {
            if value.0 == 0 {
                Err(SKError::Conversion(
                    "external categorical source empty".into(),
                ))
            } else {
                let empty: &[&[&str]] = &[];
                Ok(SKNominalMatrixView::new(empty).into())
            }
        }
    }

    let bad: Result<SKCategoricalView<'_>, SKError> = External(0).try_into();
    match bad {
        Err(SKError::Conversion(msg)) => assert!(msg.contains("empty")),
        other => panic!("expected conversion failure, got {other:?}"),
    }
}

/// The missing sentinel is reachable from the crate root and is `-1`.
#[test]
fn missing_sentinel_is_documented_constant() {
    assert_eq!(crate::SK_CATEGORICAL_MISSING_CODE, -1_i64);
    assert_eq!(SK_CATEGORICAL_MISSING_CODE, -1_i64);
}

/// The categorical view is non-empty iff its matrix has rows and columns.
#[test]
fn categorical_view_emptiness() {
    let codes = Array2::from_shape_vec((0, 0), Vec::<i64>::new()).unwrap();
    let view: SKCategoricalView<'_> = ArrayView2::from(&codes).into();
    assert!(view.is_empty());
    assert_eq!(view.number_of_columns(), 0);
}
