//! TDD tests for the layout helpers (spec `layout`).

use ndarray::{Array, Array2, ShapeBuilder, array, s};

use super::{
    SKMemoryLayout, sk_force_contiguous, sk_is_c_contiguous, sk_is_f_contiguous, sk_memory_layout,
};

/// A row-major owned array is detected as C-contiguous.
#[test]
fn row_major_array_is_c_contiguous() {
    let input: Array2<f64> = array![[1.0, 2.0], [3.0, 4.0]];
    assert!(sk_is_c_contiguous(&input.view()));
    assert!(!sk_is_f_contiguous(&input.view()));
    assert_eq!(sk_memory_layout(&input.view()), SKMemoryLayout::CContiguous);
}

/// An F-order array is detected as F-contiguous.
#[test]
fn column_major_array_is_f_contiguous() {
    let f_order = Array::from_shape_vec((2, 2).f(), vec![1.0_f64, 2.0, 3.0, 4.0]).unwrap();
    assert!(sk_is_f_contiguous(&f_order.view()));
    assert_eq!(
        sk_memory_layout(&f_order.view()),
        SKMemoryLayout::FContiguous
    );
}

/// A strided view (every other column) is detected as strided.
#[test]
fn strided_view_is_detected() {
    let base: Array2<f64> = array![[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]];
    let strided = base.slice(s![.., ..;2]);
    assert!(!strided.is_standard_layout());
    assert!(!strided.t().is_standard_layout());
    assert_eq!(sk_memory_layout(&strided), SKMemoryLayout::Strided);
    assert!(!sk_is_c_contiguous(&strided));
    assert!(!sk_is_f_contiguous(&strided));
}

/// Forcing contiguity on a strided view yields a contiguous owned copy.
#[test]
fn force_contiguity_owns_when_strided() {
    let base: Array2<f64> = array![[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]];
    let strided = base.slice(s![.., ..;2]); // (2, 2) strided view.
    let forced = sk_force_contiguous(&strided);
    assert!(forced.is_standard_layout());
    // Values are preserved element-wise.
    assert_eq!(forced[[0, 0]], strided[[0, 0]]);
    assert_eq!(forced[[1, 1]], strided[[1, 1]]);
}

/// The layout enum covers exactly the three standard cases.
#[test]
fn layout_enum_exhaustive_over_standard_cases() {
    let c: Array2<f64> = array![[1.0, 2.0], [3.0, 4.0]];
    let f: Array2<f64> = c.t().to_owned();
    assert!(matches!(
        sk_memory_layout(&c.view()),
        SKMemoryLayout::CContiguous
    ));
    assert!(matches!(
        sk_memory_layout(&f.view()),
        SKMemoryLayout::FContiguous
    ));
}
