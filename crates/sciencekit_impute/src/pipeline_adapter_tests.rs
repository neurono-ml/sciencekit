//! Tests for the impute outputs' entry into the pipeline block medium
//! (spec `typed-pipeline`).

use ndarray::Array2;
use sprs::CsMat;

use crate::missing_indicator::SKIndicatorOutput;
use crate::simple_imputer::SKImputedOutput;
use sciencekit_common::{SKFeatureBlock, SKIntoFeatureBlock};

fn dense_matrix() -> Array2<f64> {
    Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap()
}

fn sparse_matrix() -> CsMat<f64> {
    CsMat::new((2, 3), vec![0, 1, 2], vec![0, 2], vec![1.0, 2.0])
}

#[test]
fn imputed_dense_output_converts_to_a_dense_block() {
    let block = SKImputedOutput::Dense(dense_matrix()).into_feature_block();
    assert!(!block.is_sparse());
    assert_eq!(block.number_of_rows(), 2);
}

#[test]
fn imputed_sparse_output_converts_to_a_sparse_block() {
    let block = SKImputedOutput::Sparse(sparse_matrix()).into_feature_block();
    assert!(block.is_sparse());
    assert_eq!(block.number_of_columns(), 3);
}

#[test]
fn indicator_dense_output_converts_to_a_dense_block() {
    let block = SKIndicatorOutput::Dense(dense_matrix()).into_feature_block();
    assert!(!block.is_sparse());
}

#[test]
fn indicator_sparse_output_converts_to_a_sparse_block() {
    let block = SKIndicatorOutput::Sparse(sparse_matrix()).into_feature_block();
    assert!(block.is_sparse());
    assert_eq!(block.number_of_rows(), 2);
}

#[test]
fn f32_outputs_convert_like_f64_outputs() {
    let dense = Array2::from_shape_vec((1, 2), vec![1.0_f32, 2.0]).unwrap();
    let block: SKFeatureBlock<f32> = SKImputedOutput::Dense(dense).into_feature_block();
    assert!(!block.is_sparse());
}
