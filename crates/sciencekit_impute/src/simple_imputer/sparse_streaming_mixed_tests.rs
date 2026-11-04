//! Mixed dense and sparse coverage for the streaming simple imputer.

use ndarray::array;
use sciencekit_common::{SKBuilder, SKDataBatch, SKUnsupervisedFit};

use super::core_implementation::SKSimpleImputer;
use super::strategy::SKImputationStrategy;
use crate::sparse_streaming_test_datasets::{OwnedBatchSource, assert_close_sparse, dense_to_csr};

/// A stream that mixes one dense batch and one sparse batch learns the same
/// statistics as the dense `fit` on the concatenated data.
#[test]
fn mixed_dense_and_sparse_stream_matches_the_dense_fit() {
    let features = array![
        [1.0, 0.0, 2.0],
        [0.0, 3.0, f64::NAN],
        [4.0, 0.0, 6.0],
        [0.0, 5.0, 8.0],
        [7.0, 0.0, 0.0],
        [0.0, 9.0, 12.0],
    ];
    let reference = SKSimpleImputer::<f64>::build()
        .strategy(SKImputationStrategy::Mean)
        .build()
        .unwrap()
        .fit(&features)
        .unwrap();

    let dense_chunk = features.slice(ndarray::s![0..2, ..]).to_owned();
    let sparse_chunk = dense_to_csr(&features.slice(ndarray::s![2..6, ..]).to_owned());
    let mut source = OwnedBatchSource::new(vec![
        SKDataBatch::new_dense(dense_chunk, 0, false),
        SKDataBatch::new_sparse(sparse_chunk, 1, true),
    ]);
    let streamed = SKSimpleImputer::<f64>::build()
        .strategy(SKImputationStrategy::Mean)
        .build()
        .unwrap()
        .fit_streaming(&mut source)
        .unwrap();

    assert_eq!(
        streamed.number_of_input_columns(),
        reference.number_of_input_columns()
    );
    assert_close_sparse(
        &streamed.statistics().to_vec(),
        &reference.statistics().to_vec(),
        1e-12,
    );
}
