//! TDD suite for primitive zero-copy import (task 1.5).

use arrow_array::{Float32Array, Float64Array, Int64Array};
use arrow_buffer::NullBuffer;

use super::zero_copy::{sk_arrow_borrowed_column, sk_arrow_conversion_report};
use sciencekit_common::SKError;

#[test]
fn sk_complete_f64_column_borrows_the_arrow_buffer() {
    let array = Float64Array::from(vec![1.0, 2.0, 3.0]);

    let borrowed = sk_arrow_borrowed_column(&array).expect("a borrowable column");
    let view = borrowed.as_f64_column_view().expect("an f64 view");

    assert_eq!(view.number_of_rows(), 3);
    assert_eq!(view.matrix_view().as_ptr(), array.values().as_ptr());
    assert_eq!(
        view.matrix_view().as_slice().expect("contiguous"),
        &[1.0, 2.0, 3.0]
    );
}

#[test]
fn sk_complete_f32_and_i64_columns_borrow_the_arrow_buffer() {
    let single = Float32Array::from(vec![0.5_f32, 1.5]);
    let borrowed = sk_arrow_borrowed_column(&single).expect("a borrowable column");
    let view = borrowed.as_f32_column_view().expect("an f32 view");
    assert_eq!(view.matrix_view().as_ptr(), single.values().as_ptr());

    let counts = Int64Array::from(vec![7_i64, 8]);
    let borrowed = sk_arrow_borrowed_column(&counts).expect("a borrowable column");
    let view = borrowed.as_i64_column_view().expect("an i64 view");
    assert_eq!(view.matrix_view().as_ptr(), counts.values().as_ptr());
}

#[test]
fn sk_nullable_column_is_not_borrowed() {
    let array = Float64Array::new(
        vec![1.0, 2.0, 3.0].into(),
        Some(NullBuffer::from(vec![true, false, true])),
    );

    let error = sk_arrow_borrowed_column(&array).expect_err("nulls force materialization");
    assert!(
        matches!(error, SKError::UnsupportedRepresentation { .. }),
        "unexpected error: {error}"
    );
}

#[test]
fn sk_conversion_report_distinguishes_zero_copy_from_cast_and_materialization() {
    let complete = Float64Array::from(vec![1.0, 2.0]);
    assert!(sk_arrow_conversion_report::<f64>(&complete).is_zero_copy());
    assert!(!sk_arrow_conversion_report::<f64>(&complete).is_cast());

    let single = Float32Array::from(vec![1.0_f32, 2.0]);
    let cast = sk_arrow_conversion_report::<f64>(&single);
    assert!(!cast.is_zero_copy());
    assert!(cast.is_cast());

    let nullable = Float64Array::new(
        vec![1.0, 2.0].into(),
        Some(NullBuffer::from(vec![true, false])),
    );
    let materialized = sk_arrow_conversion_report::<f64>(&nullable);
    assert!(!materialized.is_zero_copy());
    assert!(!materialized.is_cast());
}
