//! Companion TDD tests for owned support-vector blocks (spec
//! `support-vector-machines`).

use ndarray::array;
use sciencekit_common::SKDataView;
use sprs::CsMat;

use super::SKOwnedFeatureBlock;

#[test]
fn owned_feature_block_preserves_csr_representation_without_densifying() {
    let csr = CsMat::new(
        (4, 50_000_000),
        vec![0, 1, 3, 3, 4],
        vec![7, 12, 49_999_999, 100],
        vec![2.0_f64, -3.0, 5.0, 1.5],
    );
    let block =
        SKOwnedFeatureBlock::select_rows(SKDataView::Sparse(csr.view()), &[1, 3]).unwrap();
    assert!(block.is_sparse());
    assert_eq!(block.rows(), 2);
    assert_eq!(block.columns(), 50_000_000);

    let dense = array![[1.0_f64, 2.0], [3.0, 4.0], [5.0, 6.0]];
    let dense_block =
        SKOwnedFeatureBlock::select_rows(SKDataView::Dense(dense.view()), &[0, 2]).unwrap();
    assert!(!dense_block.is_sparse());
    assert_eq!(dense_block.rows(), 2);
}
