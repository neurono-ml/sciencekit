//! Output representation, float and scale tests for `SKPolynomialFeatures`
//! (spec `polynomial-features`).

use ndarray::{Array2, array};
use sciencekit_common::SKBuilder;

use super::core_implementation::SKPolynomialFeatures;
use super::output::SKExpandedOutput;
use super::polynomial_features_tests::default_expander;

/// Both supported scalars work.
#[test]
fn both_supported_floats() {
    let features = Array2::from_shape_vec((1, 2), vec![2.0_f64, 3.0]).unwrap();
    let model = default_expander().fit(features.view()).unwrap();
    let as_f32 = model
        .transform_dense::<f32, _>(
            Array2::from_shape_vec((1, 2), vec![2.0_f32, 3.0])
                .unwrap()
                .view(),
        )
        .unwrap();
    let as_f64: Array2<f64> = model.transform_dense(features.view()).unwrap();
    assert_eq!(as_f32[[0, 5]], 9.0_f32);
    assert_eq!(as_f64[[0, 5]], 9.0_f64);
}

/// Sparse output equals the dense output.
#[test]
fn sparse_matches_dense() {
    let features = array![[2.0_f64, 3.0], [0.0, 4.0]];
    let model = default_expander().fit(features.view()).unwrap();
    let dense: Array2<f64> = model.transform_dense(features.view()).unwrap();
    let sparse = model.transform_sparse::<f64, _>(features.view()).unwrap();
    assert_eq!(sparse.to_dense(), dense);
}

/// The builder selects the default output representation.
#[test]
fn builder_selects_default_output() {
    let features = array![[2.0_f64, 3.0]];
    let sparse_model = SKPolynomialFeatures::build()
        .sparse_output(true)
        .build()
        .unwrap()
        .fit(features.view())
        .unwrap();
    match sparse_model.transform::<f64, _>(features.view()).unwrap() {
        SKExpandedOutput::Sparse(_) => {}
        other => panic!("expected sparse, got {other:?}"),
    }

    let dense_model = default_expander().fit(features.view()).unwrap();
    match dense_model.transform::<f64, _>(features.view()).unwrap() {
        SKExpandedOutput::Dense(_) => {}
        other => panic!("expected dense, got {other:?}"),
    }
}

/// Acceptance: large data expands, and dense and sparse paths agree.
#[test]
fn large_data_dense_and_sparse_agree() {
    let features = Array2::from_shape_fn((20_000, 5), |(row, column)| {
        ((row * 7 + column * 13) % 17) as f64 * 0.5 - 2.0
    });
    let model = default_expander().fit(features.view()).unwrap();
    // Two features short of the count: degree-2 expansion of 5 features is
    // C(5 + 2, 2) = 21 terms (with bias).
    assert_eq!(model.number_of_output_features(), 21);

    let dense: Array2<f64> = model.transform_dense(features.view()).unwrap();
    let sparse = model.transform_sparse::<f64, _>(features.view()).unwrap();
    assert_eq!(sparse.to_dense(), dense);
}

/// The default execution intent is Automatic.
#[test]
fn default_execution_intent_is_automatic() {
    use sciencekit_common::SKExecutionMode;
    let expander = SKPolynomialFeatures::build().build().unwrap();
    assert_eq!(expander.execution_intent(), SKExecutionMode::Automatic);
}

/// Zero terms are not stored in the sparse output.
#[test]
fn sparse_stores_only_nonzeros() {
    let features = array![[2.0_f64, 3.0], [0.0, 4.0]];
    let model = default_expander().fit(features.view()).unwrap();
    let sparse = model.transform_sparse::<f64, _>(features.view()).unwrap();
    // Row 0: [1,2,3,4,6,9] -> 6 nonzeros. Row 1: [1,0,4,0,0,16] -> 3 nonzeros.
    assert_eq!(sparse.nnz(), 9);
}
