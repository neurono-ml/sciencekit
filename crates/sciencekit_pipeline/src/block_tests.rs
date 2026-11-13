//! Tests for the pipeline medium re-export.

use ndarray::Array2;
use sprs::CsMat;

use super::{SKFeatureBlock, SKIntoFeatureBlock};

#[test]
fn dense_adapter_preserves_the_dense_representation() {
    let matrix = Array2::from_shape_vec((2, 2), vec![1.0_f64, 2.0, 3.0, 4.0]).unwrap();
    let block = SKFeatureBlock::from(matrix);
    assert!(!block.is_sparse());
    assert_eq!(block.number_of_rows(), 2);
    assert_eq!(block.number_of_columns(), 2);
}

#[test]
fn sparse_adapter_preserves_the_csr_representation() {
    let matrix = CsMat::new((2, 3), vec![0, 1, 2], vec![0, 2], vec![1.0_f64, 2.0]);
    let block = SKFeatureBlock::from(matrix);
    assert!(block.is_sparse());
    assert_eq!(block.number_of_columns(), 3);
}

#[test]
fn external_outputs_convert_through_the_pipeline_trait() {
    let dense = Array2::from_shape_vec((1, 2), vec![1.0_f64, 2.0])
        .unwrap()
        .into_feature_block();
    assert!(!dense.is_sparse());
    let sparse = CsMat::new((1, 2), vec![0, 1], vec![1], vec![1.0_f64]).into_feature_block();
    assert!(sparse.is_sparse());
}
