//! Sparse streaming conformance tests for `SKLogisticRegression`.
//!
//! The retained CSR design and its canonical labels reproduce the dense stream
//! and the dense `fit`. The guarded test retains a CSR design at a shape whose
//! dense twin is impractical for the test budget. It compares against the dense
//! `fit` only; no dense stream of the same data is exercised here.

use ndarray::Array1;
use sciencekit_common::SKSupervisedFit;
use sciencekit_common::builders::SKBuilder;

use super::SKLogisticRegression;
use crate::logistic::test_datasets::{labels_i64, separable_binary_design};
use crate::sparse_streaming_test_datasets::{
    SparseSupervisedSource, chunk_pairs_csr, guard_matrix,
};

fn label_batches(raw: &[usize]) -> Array1<f64> {
    Array1::from_vec(raw.iter().map(|&label| label as f64).collect())
}

#[test]
fn sparse_stream_matches_dense_fit() {
    let (features, raw) = separable_binary_design();
    let labels = labels_i64(&raw);
    let reference = SKLogisticRegression::<f64>::build()
        .build()
        .unwrap()
        .fit(features.view(), labels.view())
        .unwrap();
    let targets = label_batches(&raw);
    let streamed = SKLogisticRegression::<f64>::build()
        .build()
        .unwrap()
        .fit_streaming(&mut SparseSupervisedSource::new(chunk_pairs_csr(
            &features,
            &targets,
            &[4, 2],
        )))
        .unwrap();
    for (dense_value, streamed_value) in
        reference.coefficients().iter().zip(streamed.coefficients())
    {
        assert!((dense_value - streamed_value).abs() < 1e-6);
    }
    assert_eq!(streamed.row_count(), reference.row_count());
}

/// The retained design streams at a shape whose dense twin is impractical for
/// the test budget.
#[test]
fn sparse_stream_handles_a_shape_beyond_the_test_budget() {
    let rows = 100_000;
    let labels = Array1::from_shape_fn(rows, |row| (row % 2) as f64);
    let mut source = SparseSupervisedSource::<f64>::new(vec![(guard_matrix(rows, 32), labels)]);
    let streamed = SKLogisticRegression::<f64>::build()
        .maximum_number_of_iterations(10)
        .build()
        .unwrap()
        .fit_streaming(&mut source)
        .unwrap();
    assert_eq!(streamed.feature_count(), 32);
    assert_eq!(streamed.row_count(), rows as u64);
}
