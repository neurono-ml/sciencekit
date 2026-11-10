//! Tests for scalar coercion of supported columns (spec
//! `dataset-interop-core`).

use sciencekit_common::SKError;

use super::{sk_coerce_numeric_column, sk_integer_is_representable};
use crate::columns::SKColumnData;
use crate::missing::SKMissingValuePolicy;

/// Every supported scalar column type lands in the model scalar.
#[test]
fn supported_columns_are_coerced() {
    let policy = SKMissingValuePolicy::Reject;

    let floats32 = sk_coerce_numeric_column::<f64>(
        "single",
        &SKColumnData::Float32(vec![Some(0.5), Some(1.5)]),
        policy,
    )
    .unwrap();
    assert_eq!(floats32.to_vec(), vec![0.5, 1.5]);

    let floats64 = sk_coerce_numeric_column::<f32>(
        "double",
        &SKColumnData::Float64(vec![Some(2.5), Some(-1.0)]),
        policy,
    )
    .unwrap();
    assert_eq!(floats64.to_vec(), vec![2.5, -1.0]);

    let integers = sk_coerce_numeric_column::<f64>(
        "count",
        &SKColumnData::Integer64(vec![Some(1), Some(-2), Some(3)]),
        policy,
    )
    .unwrap();
    assert_eq!(integers.to_vec(), vec![1.0, -2.0, 3.0]);

    let booleans = sk_coerce_numeric_column::<f64>(
        "flag",
        &SKColumnData::Boolean(vec![Some(true), Some(false), Some(true)]),
        policy,
    )
    .unwrap();
    assert_eq!(booleans.to_vec(), vec![1.0, 0.0, 1.0]);
}

/// Integer magnitudes are exact only within the scalar mantissa; the guard
/// reports the bound honestly.
#[test]
fn integer_representability_follows_the_mantissa() {
    assert!(sk_integer_is_representable::<f64>(0));
    assert!(sk_integer_is_representable::<f64>(1 << 53));
    assert!(!sk_integer_is_representable::<f64>((1 << 53) + 1));
    assert!(sk_integer_is_representable::<f32>(1 << 24));
    assert!(!sk_integer_is_representable::<f32>((1 << 24) + 1));
}

/// The reject policy stops on the first null and names the column.
#[test]
fn reject_policy_stops_on_a_null() {
    let error = sk_coerce_numeric_column::<f64>(
        "price",
        &SKColumnData::Float64(vec![Some(1.0), None]),
        SKMissingValuePolicy::Reject,
    )
    .unwrap_err();
    match &error {
        SKError::NullValueInColumn { column, policy } => {
            assert_eq!(column, "price");
            assert_eq!(*policy, "reject");
        }
        other => panic!("expected a null-in-column error, got {other:?}"),
    }
    assert!(error.to_string().contains("price"));
}

/// The numeric policy writes the NaN sentinel at the null position.
#[test]
fn numeric_policy_maps_null_to_nan() {
    let values = sk_coerce_numeric_column::<f64>(
        "price",
        &SKColumnData::Float64(vec![Some(1.0), None, Some(3.0)]),
        SKMissingValuePolicy::NumericNaN,
    )
    .unwrap();
    assert_eq!(values.len(), 3);
    assert_eq!(values[0], 1.0);
    assert!(values[1].is_nan());
    assert_eq!(values[2], 3.0);
}

/// Constant fill replaces nulls with the validated constant.
#[test]
fn constant_fill_replaces_nulls() {
    let values = sk_coerce_numeric_column::<f32>(
        "price",
        &SKColumnData::Float32(vec![None, Some(2.0)]),
        SKMissingValuePolicy::constant_fill(-1.5).unwrap(),
    )
    .unwrap();
    assert_eq!(values.to_vec(), vec![-1.5, 2.0]);
}

/// An integer column with a null is promoted before the sentinel is written,
/// never truncated.
#[test]
fn integer_null_promotion_is_not_truncated() {
    let values = sk_coerce_numeric_column::<f64>(
        "count",
        &SKColumnData::Integer64(vec![Some(7), None]),
        SKMissingValuePolicy::NumericNaN,
    )
    .unwrap();
    assert_eq!(values[0], 7.0);
    assert!(values[1].is_nan());
}

/// An integer above the scalar's exact range is rejected during promotion.
#[test]
fn integer_null_promotion_rejects_unrepresentable_magnitudes() {
    let error = sk_coerce_numeric_column::<f64>(
        "identifier",
        &SKColumnData::Integer64(vec![Some(i64::MAX), None]),
        SKMissingValuePolicy::NumericNaN,
    )
    .unwrap_err();
    match &error {
        SKError::Conversion(message) => assert!(message.contains("identifier")),
        other => panic!("expected a conversion error, got {other:?}"),
    }
}

/// The categorical policy cannot handle numeric nulls; the engine says so.
#[test]
fn categorical_policy_rejects_numeric_nulls() {
    let error = sk_coerce_numeric_column::<f64>(
        "price",
        &SKColumnData::Float64(vec![None]),
        SKMissingValuePolicy::CategoricalMissingCode,
    )
    .unwrap_err();
    assert!(matches!(error, SKError::NullValueInColumn { .. }));
}

/// A nominal column is unsupported on the numeric path, and the suggestion
/// names the conversion.
#[test]
fn nominal_columns_are_unsupported_on_the_numeric_path() {
    let error = sk_coerce_numeric_column::<f64>(
        "species",
        &SKColumnData::Nominal(vec![Some("cat".into())]),
        SKMissingValuePolicy::Reject,
    )
    .unwrap_err();
    match &error {
        SKError::UnsupportedColumnType {
            column,
            logical_type,
            suggestion,
        } => {
            assert_eq!(column, "species");
            assert_eq!(logical_type, "nominal");
            assert!(suggestion.contains("category"));
        }
        other => panic!("expected an unsupported-column-type error, got {other:?}"),
    }
}

/// Category codes take the categorical path, not the numeric one.
#[test]
fn category_codes_are_unsupported_on_the_numeric_path() {
    let error = sk_coerce_numeric_column::<f64>(
        "species_code",
        &SKColumnData::CategoryCode(vec![Some(0)]),
        SKMissingValuePolicy::Reject,
    )
    .unwrap_err();
    assert!(matches!(error, SKError::UnsupportedColumnType { .. }));
}
