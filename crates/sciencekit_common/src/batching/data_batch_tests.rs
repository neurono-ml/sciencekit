//! Tests for the representation-aware streaming data batch (spec
//! `sparse-streaming-batches`).

use ndarray::Array2;
use sprs::CsMat;

use super::SKDataBatch;
use crate::SKDataView;

fn dense_batch() -> SKDataBatch<f64> {
    let data = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    SKDataBatch::new_dense(data, 4, true)
}

fn sparse_batch() -> SKDataBatch<f64> {
    let data = CsMat::new((2, 4), vec![0, 1, 2], vec![1, 3], vec![10.0_f64, 20.0]);
    SKDataBatch::new_sparse(data, 7, false)
}

/// The dense constructor keeps its meaning and preserves position and flag.
#[test]
fn dense_batch_preserves_metadata() {
    let batch = dense_batch();
    assert_eq!(batch.position(), 4);
    assert!(batch.is_final());
    assert_eq!(batch.number_of_rows(), 2);
    assert_eq!(batch.number_of_columns(), 3);
    assert!(!batch.is_sparse());
    match batch.view() {
        SKDataView::Dense(view) => assert_eq!(view[[1, 2]], 6.0),
        SKDataView::Sparse(_) => panic!("expected a dense view"),
    }
}

/// `new` stays the dense alias of `new_dense`.
#[test]
fn dense_alias_equals_new_dense() {
    let data = Array2::from_shape_vec((1, 2), vec![1.0_f64, 2.0]).unwrap();
    let batch = SKDataBatch::new(data, 0, false);
    assert!(!batch.is_sparse());
    assert_eq!(batch.number_of_rows(), 1);
    assert_eq!(batch.number_of_columns(), 2);
    match batch.view() {
        SKDataView::Dense(view) => assert_eq!(view, ndarray::array![[1.0_f64, 2.0]]),
        SKDataView::Sparse(_) => panic!("expected a dense view"),
    }
}

#[test]
fn sparse_batch_preserves_metadata_and_representation() {
    let batch = sparse_batch();
    assert_eq!(batch.position(), 7);
    assert!(!batch.is_final());
    assert_eq!(batch.number_of_rows(), 2);
    assert_eq!(batch.number_of_columns(), 4);
    assert!(batch.is_sparse());
    match batch.view() {
        SKDataView::Sparse(view) => assert_eq!(view.nnz(), 2),
        SKDataView::Dense(_) => panic!("expected a sparse view"),
    }
}

/// Batches carry f32 as well as f64 without changing the surface.
#[test]
fn batch_supports_f32() {
    let data = Array2::from_shape_vec((1, 2), vec![1.0_f32, 2.0]).unwrap();
    let batch = SKDataBatch::new_dense(data, 3, true);
    assert_eq!(batch.number_of_rows(), 1);
    assert_eq!(batch.number_of_columns(), 2);
    assert!(!batch.is_sparse());
}
