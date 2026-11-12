//! Test-transformer contract tests (spec `streaming-pipeline`, task 1.2).

use ndarray::Array2;
use sciencekit_common::SKFeatureBlock;

use crate::streaming::SKStreamingTransformer;
use crate::streaming::testing::{
    SKTestMeanCentering, sk_reopenable_blocks, sk_single_block, sk_tiny_blocks,
};

#[test]
fn a_streaming_fit_reads_the_source_and_centers_batches() {
    let transformer = SKTestMeanCentering::<f64>::new();
    let mut provider = sk_reopenable_blocks(sk_tiny_blocks::<f64>());
    let model = transformer.fit_streaming(&mut provider, 1).unwrap();
    let batch = sk_single_block(2, 2, vec![10.0, 20.0, 30.0, 40.0]);
    let transformed = transformer.transform_batch(&model, &batch).unwrap();
    let SKFeatureBlock::Dense(matrix) = transformed else {
        panic!("mean centering returns a dense block");
    };
    // Column means over 1..=12 are 6.0 and 7.0.
    assert_eq!(
        matrix,
        Array2::from_shape_vec((2, 2), vec![4.0, 13.0, 24.0, 33.0]).unwrap()
    );
}

#[test]
fn one_pass_and_two_epochs_learn_the_same_mean() {
    let transformer = SKTestMeanCentering::<f64>::new();
    let mut one_pass = sk_reopenable_blocks(sk_tiny_blocks::<f64>());
    let mut two_epochs = sk_reopenable_blocks(sk_tiny_blocks::<f64>());
    let one_pass_model = transformer.fit_streaming(&mut one_pass, 1).unwrap();
    let two_epoch_model = transformer.fit_streaming(&mut two_epochs, 2).unwrap();
    let batch = sk_single_block(1, 2, vec![6.0, 7.0]);
    let first = transformer
        .transform_batch(&one_pass_model, &batch)
        .unwrap();
    let second = transformer
        .transform_batch(&two_epoch_model, &batch)
        .unwrap();
    let (SKFeatureBlock::Dense(first), SKFeatureBlock::Dense(second)) = (first, second) else {
        panic!("mean centering returns dense blocks");
    };
    assert!(first.iter().all(|value: &f64| value.abs() < 1e-12));
    assert!(second.iter().all(|value: &f64| value.abs() < 1e-12));
}

#[test]
fn a_transformed_batch_keeps_its_row_count() {
    let transformer = SKTestMeanCentering::<f64>::new();
    let mut provider = sk_reopenable_blocks(sk_tiny_blocks::<f64>());
    let model = transformer.fit_streaming(&mut provider, 1).unwrap();
    let batch = sk_single_block(3, 2, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let transformed = transformer.transform_batch(&model, &batch).unwrap();
    assert_eq!(transformed.number_of_rows(), 3);
}
