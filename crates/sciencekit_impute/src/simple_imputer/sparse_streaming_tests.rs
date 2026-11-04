//! Sparse streaming conformance tests for `SKSimpleImputer`.
//!
//! A sparse CSR stream learns the same per-column statistics as the dense stream
//! and the dense `fit` on the concatenated data; the `0.0` marker still claims
//! the implicit zeros as missing; the missing indicator stacks after the imputed
//! block. The guarded test streams a shape whose dense twin is impractical for
//! the test budget; it proves the stream completes there, not that no dense copy
//! exists.

use ndarray::{Array2, array};
use sciencekit_common::{SKBuilder, SKExecutionMode, SKFeatureTransformer, SKUnsupervisedFit};

use super::core_implementation::SKSimpleImputer;
use super::strategy::SKImputationStrategy;
use super::streaming_tests::{SKETCH_RELATIVE_TOLERANCE, SKVectorSource, chunk_rows};
use crate::sparse_streaming_test_datasets::{
    SparseVecSource, assert_close_sparse, chunk_csr, dense_to_csr, guard_matrix,
};

/// A deterministic design with many exact zeros and a missing cell.
fn conformance_features() -> Array2<f64> {
    array![
        [1.0, 0.0, 2.0],
        [0.0, 3.0, f64::NAN],
        [4.0, 0.0, 6.0],
        [0.0, 5.0, 8.0],
        [7.0, 0.0, 0.0],
        [0.0, 9.0, 12.0],
    ]
}

fn imputer(strategy: SKImputationStrategy) -> SKSimpleImputer<f64> {
    SKSimpleImputer::<f64>::build()
        .strategy(strategy)
        .execution_mode(SKExecutionMode::OutOfCoreStreaming)
        .build()
        .expect("imputer builds")
}

/// Mean, mode and constant agree exactly across the sparse stream, the dense
/// stream and the dense fit.
#[test]
fn sparse_stream_matches_dense_for_exact_strategies() {
    let features = conformance_features();
    for strategy in [
        SKImputationStrategy::Mean,
        SKImputationStrategy::MostFrequent,
        SKImputationStrategy::Constant,
    ] {
        let reference = imputer(strategy).fit(&features).expect("dense fit");
        let dense_stream = imputer(strategy)
            .fit_streaming(&mut SKVectorSource::new(chunk_rows(&features, &[2, 1, 3])))
            .expect("dense stream");
        let sparse_stream = imputer(strategy)
            .fit_streaming(&mut SparseVecSource::new(chunk_csr(&features, &[2, 1, 3])))
            .expect("sparse stream");
        assert_close_sparse(
            &sparse_stream.statistics().to_vec(),
            &reference.statistics().to_vec(),
            1e-12,
        );
        assert_close_sparse(
            &sparse_stream.statistics().to_vec(),
            &dense_stream.statistics().to_vec(),
            1e-12,
        );
    }
}

/// The median sketch of a sparse stream approximates the dense fit within the
/// documented tolerance.
#[test]
fn sparse_stream_median_matches_dense_within_the_sketch_tolerance() {
    let features = Array2::from_shape_fn((600, 2), |(row, column)| {
        if (row + column) % 5 == 0 {
            0.0
        } else {
            ((row * 7 + column * 13) % 71) as f64 * 0.5
        }
    });
    let strategy = SKImputationStrategy::Median;
    let reference = imputer(strategy).fit(&features).expect("dense fit");
    let sparse_stream = imputer(strategy)
        .fit_streaming(&mut SparseVecSource::new(chunk_csr(&features, &[200, 400])))
        .expect("sparse stream");
    assert_close_sparse(
        &sparse_stream.statistics().to_vec(),
        &reference.statistics().to_vec(),
        SKETCH_RELATIVE_TOLERANCE,
    );
}

/// The `0.0` marker claims implicit zeros: a sparse column with no stored entry
/// is entirely missing and every column joins the indicator scope.
#[test]
fn zero_marker_claims_implicit_zeros() {
    let features = array![[1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0]];
    let reference = SKSimpleImputer::<f64>::build()
        .missing_values(0.0)
        .add_indicator(true)
        .build()
        .unwrap()
        .fit(&features)
        .unwrap();
    let streamed = SKSimpleImputer::<f64>::build()
        .missing_values(0.0)
        .add_indicator(true)
        .execution_mode(SKExecutionMode::OutOfCoreStreaming)
        .build()
        .unwrap()
        .fit_streaming(&mut SparseVecSource::new(chunk_csr(&features, &[1, 3])))
        .unwrap();
    let streamed_statistics = streamed.statistics();
    let reference_statistics = reference.statistics();
    for (actual, expected) in streamed_statistics.iter().zip(reference_statistics.iter()) {
        assert_eq!(actual.to_bits(), expected.to_bits());
    }
    assert_eq!(
        streamed.number_of_output_features(),
        reference.number_of_output_features()
    );
    assert!(streamed.indicator().is_some());
}

/// The missing indicator stacks after the imputed block for a sparse transform.
#[test]
fn add_indicator_stacks_after_the_imputed_block() {
    let features = conformance_features();
    let reference = SKSimpleImputer::<f64>::build()
        .add_indicator(true)
        .build()
        .unwrap()
        .fit(&features)
        .unwrap();
    let streamed = SKSimpleImputer::<f64>::build()
        .add_indicator(true)
        .execution_mode(SKExecutionMode::OutOfCoreStreaming)
        .build()
        .unwrap()
        .fit_streaming(&mut SparseVecSource::new(chunk_csr(&features, &[2, 4])))
        .unwrap();
    assert_eq!(
        streamed.number_of_output_features(),
        reference.number_of_output_features()
    );
    let sparse = dense_to_csr(&features);
    let expected = reference.transform(sparse.view()).unwrap();
    let actual = streamed.transform(sparse.view()).unwrap();
    assert_eq!(actual, expected);
}

/// The stream completes at a shape whose dense twin is impractical for the test
/// budget.
#[test]
fn sparse_stream_handles_a_shape_beyond_the_test_budget() {
    let batches = vec![
        guard_matrix(700_000, 256),
        guard_matrix(700_000, 256),
        guard_matrix(700_000, 256),
    ];
    let streamed = imputer(SKImputationStrategy::Mean)
        .fit_streaming(&mut SparseVecSource::new(batches))
        .expect("sparse stream");
    assert_eq!(streamed.number_of_input_columns(), 256);
    assert_eq!(streamed.statistics().len(), 256);
}
