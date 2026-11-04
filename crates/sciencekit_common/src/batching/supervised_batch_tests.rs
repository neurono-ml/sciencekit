//! Tests for the representation-aware supervised streaming batch (spec
//! `sparse-streaming-batches`).

use ndarray::{Array1, Array2};
use sprs::CsMat;

use super::SKSupervisedBatch;
use crate::{SKDataView, SKError};

#[test]
fn dense_supervised_batch_pairs_features_and_targets() {
    let features = Array2::from_shape_vec((2, 2), vec![1.0_f64, 2.0, 3.0, 4.0]).unwrap();
    let targets = Array1::from_vec(vec![5.0_f64, 6.0]);
    let batch = SKSupervisedBatch::new_dense(features, targets, 1, true).unwrap();

    assert_eq!(batch.position(), 1);
    assert!(batch.is_final());
    assert_eq!(batch.number_of_rows(), 2);
    assert_eq!(batch.number_of_columns(), 2);
    assert!(!batch.is_sparse());
    assert_eq!(batch.targets().as_slice().unwrap(), &[5.0, 6.0]);
}

#[test]
fn sparse_supervised_batch_exposes_the_sparse_view() {
    let features = CsMat::new((2, 4), vec![0, 1, 2], vec![1, 3], vec![10.0_f64, 20.0]);
    let targets = Array1::from_vec(vec![1.0_f64, 2.0]);
    let batch = SKSupervisedBatch::new_sparse(features, targets, 2, false).unwrap();

    assert!(batch.is_sparse());
    assert_eq!(batch.number_of_rows(), 2);
    assert_eq!(batch.number_of_columns(), 4);
    assert_eq!(batch.targets().as_slice().unwrap(), &[1.0, 2.0]);
    match batch.view() {
        SKDataView::Sparse(view) => assert_eq!(view.nnz(), 2),
        SKDataView::Dense(_) => panic!("expected a sparse view"),
    }
}

#[test]
fn dense_alias_equals_new_dense() {
    let features = Array2::zeros((1, 1));
    let targets = Array1::from_vec(vec![3.0_f64]);
    let batch = SKSupervisedBatch::new(features, targets, 0, true).unwrap();
    assert_eq!(batch.number_of_rows(), 1);
    assert_eq!(batch.targets()[0], 3.0);
}

#[test]
fn target_count_mismatch_is_rejected() {
    let features = Array2::zeros((3, 2));
    let targets = Array1::from_vec(vec![1.0_f64, 2.0]);
    match SKSupervisedBatch::new_dense(features, targets, 0, false) {
        Err(SKError::ShapeMismatch { expected, found }) => {
            assert_eq!(expected, vec![3]);
            assert_eq!(found, vec![2]);
        }
        other => panic!("expected a shape mismatch, got {other:?}"),
    }
}

#[test]
fn sparse_target_count_mismatch_is_rejected() {
    let features = CsMat::new((2, 2), vec![0, 1, 2], vec![0, 1], vec![1.0_f64, 2.0]);
    let targets = Array1::from_vec(vec![1.0_f64, 2.0, 3.0]);
    assert!(SKSupervisedBatch::new_sparse(features, targets, 0, false).is_err());
}

/// A sparse supervised batch survives a move to another thread unchanged.
#[test]
fn sparse_supervised_batch_moves_across_threads() {
    let features = CsMat::new((2, 4), vec![0, 1, 2], vec![1, 3], vec![10.0_f64, 20.0]);
    let targets = Array1::from_vec(vec![1.0_f64, 2.0]);
    let batch = SKSupervisedBatch::new_sparse(features, targets, 9, true).unwrap();

    let handle = std::thread::spawn(move || {
        assert_eq!(batch.position(), 9);
        assert!(batch.is_final());
        assert!(batch.is_sparse());
        match batch.view() {
            SKDataView::Sparse(view) => view.nnz(),
            SKDataView::Dense(_) => panic!("the representation must survive the move"),
        }
    });
    assert_eq!(handle.join().unwrap(), 2);
}

/// Supervised batches carry f32 as well as f64.
#[test]
fn supervised_batch_supports_f32() {
    let features = Array2::from_shape_vec((1, 2), vec![1.0_f32, 2.0]).unwrap();
    let targets = Array1::from_vec(vec![3.0_f32]);
    let batch = SKSupervisedBatch::new(features, targets, 0, true).unwrap();
    assert_eq!(batch.number_of_rows(), 1);
    assert_eq!(batch.number_of_columns(), 2);
}
