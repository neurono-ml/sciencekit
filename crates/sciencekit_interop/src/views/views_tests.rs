//! Tests for the zero-copy feature views (spec `dataset-interop-core`).

use ndarray::array;
use sciencekit_common::{SKDataView, SKError};

use super::{SKNumericColumnView, SKRowMajorFeatureView};

/// A contiguous row-major matrix is borrowed, not copied.
#[test]
fn row_major_matrix_is_borrowed() {
    let matrix = array![[1.0_f64, 2.0], [3.0, 4.0]];
    let view = SKRowMajorFeatureView::new(matrix.view()).unwrap();
    assert_eq!(view.shape(), (2, 2));

    match view.data_view() {
        SKDataView::Dense(dense) => {
            assert_eq!(dense.as_ptr(), matrix.as_ptr());
            assert_eq!(dense, matrix.view());
        }
        other => panic!("expected a dense view, got {other:?}"),
    }
}

/// A column-major (non-standard) view is rejected rather than misread.
#[test]
fn non_contiguous_view_is_rejected() {
    let matrix = array![[1.0_f64, 2.0], [3.0, 4.0]];
    let transposed = matrix.t();
    let error = SKRowMajorFeatureView::new(transposed).unwrap_err();
    assert!(matches!(error, SKError::Conversion(_)));
}

/// A single contiguous numeric column reshapes to a one-column matrix with no
/// copy.
#[test]
fn single_numeric_column_is_borrowed() {
    let values = [1.0_f32, 2.0, 3.0];
    let view = SKNumericColumnView::new(&values);
    assert_eq!(view.number_of_rows(), 3);
    assert_eq!(view.number_of_columns(), 1);

    let matrix = view.matrix_view();
    assert_eq!(matrix.dim(), (3, 1));
    assert_eq!(matrix.as_ptr(), values.as_ptr());

    match view.data_view() {
        SKDataView::Dense(dense) => {
            assert_eq!(dense.as_ptr(), values.as_ptr());
            assert_eq!(dense[[1, 0]], 2.0_f32);
        }
        other => panic!("expected a dense view, got {other:?}"),
    }
}

/// An empty column reshapes to an empty one-column matrix.
#[test]
fn empty_numeric_column_reshapes() {
    let values: [f64; 0] = [];
    let matrix = SKNumericColumnView::new(&values).matrix_view();
    assert_eq!(matrix.dim(), (0, 1));
}
