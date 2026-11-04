//! Sparse conformance tests for `SKPolynomialFeatures`.
//!
//! A sparse input expands into a sparse output in the documented column order,
//! agreeing with the dense transform; a sparse stream learns the same expansion
//! as the dense fit. The guarded test streams a tall shape whose dense twin is
//! impractical for the test budget; it proves the sparse transform completes
//! there, not that no dense copy exists.

use sciencekit_common::SKBuilder;

use super::core_implementation::SKPolynomialFeatures;
use super::polynomial_features_tests::default_expander;
use crate::sparse_streaming_test_datasets::{
    SparseVecSource, chunk_csr, dense_to_csr, guard_matrix, zeros_rich_features,
};

/// The sparse transform of a sparse input equals the dense transform, column by
/// column in the documented order.
#[test]
fn sparse_input_expands_to_the_dense_expansion() {
    let dense = zeros_rich_features();
    let sparse = dense_to_csr(&dense);
    let model = default_expander().fit(sparse.view()).expect("sparse fit");
    let sparse_output = model
        .transform_sparse(sparse.view())
        .expect("sparse transform");
    let expected = model
        .transform_dense(dense.view())
        .expect("dense transform");
    assert_eq!(sparse_output.rows(), expected.nrows());
    assert_eq!(sparse_output.cols(), expected.ncols());
    assert_eq!(sparse_output.to_dense(), expected);
}

/// A sparse stream learns the same expansion as the dense fit and expands its
/// own input sparsely.
#[test]
fn sparse_stream_matches_the_dense_fit() {
    let dense = zeros_rich_features();
    let reference = default_expander().fit(dense.view()).expect("dense fit");
    let streamed = default_expander()
        .fit_streaming(&mut SparseVecSource::new(chunk_csr(&dense, &[2, 3, 3])))
        .expect("sparse stream");
    assert_eq!(
        streamed.number_of_input_features(),
        reference.number_of_input_features()
    );
    assert_eq!(streamed.terms(), reference.terms());
    let sparse = dense_to_csr(&dense);
    let output = streamed.transform_sparse(sparse.view()).expect("transform");
    assert_eq!(
        output.to_dense(),
        reference.transform_dense(dense.view()).expect("transform")
    );
}

/// The sparse transform completes at a tall shape whose dense twin is
/// impractical for the test budget.
#[test]
fn sparse_transform_handles_a_shape_beyond_the_test_budget() {
    let rows = 2_000_000;
    let columns = 256;
    let guarded = guard_matrix(rows, columns);
    let model = SKPolynomialFeatures::build()
        .degree(1)
        .build()
        .expect("the guarded expander builds")
        .fit(guarded.view())
        .expect("sparse fit succeeds");
    let output = model
        .transform_sparse(guarded.view())
        .expect("sparse transform succeeds");
    assert_eq!(output.rows(), rows);
    assert_eq!(output.cols(), model.number_of_output_features());
    assert!(output.nnz() <= 2 * rows);
}
