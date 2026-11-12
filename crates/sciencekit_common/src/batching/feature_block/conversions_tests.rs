//! Tests for the block conversions (spec `typed-pipeline`).

use ndarray::Array2;
use sprs::CsMat;

use super::{SKFeatureBlock, SKIntoFeatureBlock};
use crate::SKDataView;

fn dense_matrix() -> Array2<f64> {
    Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap()
}

fn sparse_matrix() -> CsMat<f64> {
    CsMat::new((2, 3), vec![0, 1, 2], vec![0, 2], vec![1.0, 2.0])
}

#[test]
fn dense_matrix_becomes_a_dense_block() {
    let block = SKFeatureBlock::from(dense_matrix());
    assert!(!block.is_sparse());
    assert_eq!(block.number_of_rows(), 2);
    assert_eq!(block.number_of_columns(), 2);
}

#[test]
fn sparse_matrix_becomes_a_sparse_block() {
    let block = SKFeatureBlock::from(sparse_matrix());
    assert!(block.is_sparse());
    assert_eq!(block.number_of_columns(), 3);
}

#[test]
fn borrowed_dense_view_owns_its_copy() {
    let source = dense_matrix();
    let block = SKFeatureBlock::from(SKDataView::Dense(source.view()));
    assert!(!block.is_sparse());
    match block.view() {
        SKDataView::Dense(view) => assert_eq!(view, source.view()),
        SKDataView::Sparse(_) => panic!("dense input must stay dense"),
    }
}

#[test]
fn borrowed_sparse_view_never_densifies() {
    let mut indptr = vec![1_usize; 201];
    indptr[0] = 0;
    indptr[200] = 2;
    let source = CsMat::new((200, 400), indptr, vec![1, 399], vec![10.0_f64, 20.0]);
    let block = SKFeatureBlock::from(SKDataView::Sparse(source.view()));
    assert!(block.is_sparse());
    assert!(block.byte_size() < 200 * 400 * size_of::<f64>());
}

#[test]
fn block_borrows_back_into_the_shared_seam() {
    let block = SKFeatureBlock::from(sparse_matrix());
    let view: SKDataView<'_, f64> = SKDataView::try_from(&block).unwrap();
    assert!(matches!(view, SKDataView::Sparse(_)));
    assert_eq!(view.representation(), "csr");
}

#[test]
fn output_types_convert_into_blocks() {
    let dense = dense_matrix().into_feature_block();
    assert!(!dense.is_sparse());
    let sparse = sparse_matrix().into_feature_block();
    assert!(sparse.is_sparse());
}
