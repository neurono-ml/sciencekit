//! Tests for the data boundary (spec `data-view-boundary`).

use ndarray::{Array2, ArrayView2};
use sprs::CsMat;

use super::SKDataView;
use crate::SKError;

/// Borrowed dense matrix enters without copying — view borrows the same buffer.
#[test]
fn borrowed_dense_enters_without_copying() {
    let data = Array2::from_shape_vec((2, 2), vec![1.0_f64, 2.0, 3.0, 4.0]).unwrap();
    let view: SKDataView<'_, f64> = ArrayView2::from(&data).try_into().unwrap();
    match view {
        SKDataView::Dense(v) => {
            assert_eq!(v.as_slice_memory_order().unwrap(), &[1.0, 2.0, 3.0, 4.0]);
        }
        other => panic!("expected dense, got {other:?}"),
    }
}

/// Owned intermediate block enters by borrowing (no duplicate).
#[test]
fn owned_block_enters_by_borrowing() {
    let block = Array2::from_shape_vec((2, 1), vec![1.0_f32, 5.0]).unwrap();
    let view: SKDataView<'_, f32> = (&block).try_into().unwrap();
    match view {
        SKDataView::Dense(v) => assert_eq!(v[[1, 0]], 5.0_f32),
        other => panic!("expected dense, got {other:?}"),
    }
}

/// Sparse CSR matrix enters through the same boundary, referencing same data.
#[test]
fn sparse_enters_through_same_boundary() {
    let csr = CsMat::new_csc((2, 2), vec![0, 1, 2], vec![0, 1], vec![1.0_f64, 2.0]);
    let view: SKDataView<'_, f64> = csr.view().try_into().unwrap();
    match view {
        SKDataView::Sparse(v) => assert_eq!(v.nnz(), 2),
        other => panic!("expected sparse, got {other:?}"),
    }
}

/// A third-party type integrating via `TryFrom` is accepted by an operation
/// whose public inputs are declared over the fallible seam.
#[test]
fn user_type_with_tryfrom_works_without_friction() {
    // A third-party type that borrows an existing dense array.
    struct Wrapper<'a>(&'a Array2<f32>);
    impl<'a> TryFrom<Wrapper<'a>> for SKDataView<'a, f32> {
        type Error = SKError;
        fn try_from(w: Wrapper<'a>) -> Result<Self, Self::Error> {
            Ok(SKDataView::Dense(w.0.view()))
        }
    }

    // Generic operation whose input is declared over the fallible seam.
    fn consume<'a, X: TryInto<SKDataView<'a, f32>, Error = SKError>>(x: X) -> SKDataView<'a, f32> {
        x.try_into().unwrap()
    }
    let data = Array2::from_shape_vec((1, 2), vec![1.0, 2.0]).unwrap();
    let wrapped = Wrapper(&data);
    let _view = consume(wrapped);
}

/// A third-party fallible conversion reports a structured error via the seam.
#[test]
fn third_party_fallible_conversion_reports_structured_error() {
    struct External(u32);

    impl TryFrom<External> for SKDataView<'_, f32> {
        type Error = SKError;
        fn try_from(v: External) -> Result<Self, Self::Error> {
            if v.0 == 0 {
                Err(SKError::Conversion("external source was empty".into()))
            } else {
                // Success path: borrow a promoted empty slice (lives forever).
                let empty: &[f32] = &[];
                Ok(SKDataView::Dense(
                    ArrayView2::from_shape((0, 0), empty).unwrap(),
                ))
            }
        }
    }

    let bad: Result<SKDataView<'_, f32>, SKError> = External(0).try_into();
    match bad {
        Err(SKError::Conversion(msg)) => assert!(msg.contains("empty")),
        other => panic!("expected conversion failure, got {other:?}"),
    }
}

/// A dense-only consumer rejects a sparse input with the precise
/// unsupported-representation error before processing elements.
#[test]
fn dense_only_consumer_rejects_sparse_precisely() {
    let csr = CsMat::new_csc((2, 2), vec![0, 0, 0], vec![], Vec::<f64>::new());
    let view: SKDataView<'_, f64> = csr.view().try_into().unwrap();
    let dense_result = view.as_dense();
    match dense_result {
        Err(SKError::UnsupportedRepresentation { representation, .. }) => {
            assert_eq!(representation, "csr");
        }
        other => panic!("expected unsupported-representation, got {other:?}"),
    }
}
