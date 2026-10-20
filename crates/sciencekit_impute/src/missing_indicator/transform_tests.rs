//! Companion tests for the mask transform (spec `missing-indicator`).

use ndarray::array;
use sciencekit_common::{SKBuilder, SKError, SKFeatureTransformer, SKUnsupervisedFit};

use super::core_implementation::SKMissingIndicator;
use super::output::SKIndicatorOutput;
use super::scope::SKMissingFeatureScope;
use super::test_support::{csr, dense_mask};

#[test]
fn mask_marks_the_missing_cells() {
    let model = SKMissingIndicator::<f64>::build()
        .features(SKMissingFeatureScope::All)
        .build()
        .unwrap()
        .fit(&array![[1.0, 2.0], [3.0, 4.0]])
        .unwrap();
    let mask = dense_mask(
        model
            .transform(&array![[f64::NAN, 1.0], [2.0, f64::NAN]])
            .unwrap(),
    );
    assert_eq!(mask, array![[1.0, 0.0], [0.0, 1.0]]);
}

#[test]
fn dense_input_yields_a_dense_mask_in_scope_order() {
    let model = SKMissingIndicator::<f64>::build()
        .build()
        .unwrap()
        .fit(&array![[1.0, f64::NAN], [3.0, 4.0]])
        .unwrap();
    let mask = dense_mask(
        model
            .transform(&array![[2.0, f64::NAN], [4.0, 5.0]])
            .unwrap(),
    );
    assert_eq!(mask, array![[1.0], [0.0]]);
}

#[test]
fn a_custom_marker_marks_by_equality() {
    let model = SKMissingIndicator::<f64>::build()
        .missing_values(-1.0)
        .features(SKMissingFeatureScope::All)
        .build()
        .unwrap()
        .fit(&array![[1.0, -1.0], [f64::NAN, 4.0]])
        .unwrap();
    let mask = dense_mask(
        model
            .transform(&array![[2.0, -1.0], [f64::NAN, 5.0]])
            .unwrap(),
    );
    assert_eq!(
        mask,
        array![[0.0, 1.0], [0.0, 0.0]],
        "only the equality marker is missing, a NaN cell is observed"
    );
}

#[test]
fn sparse_input_yields_a_sparse_mask() {
    let fitted = csr((2, 2), &[(0, 0, f64::NAN), (1, 0, 3.0), (1, 1, 4.0)]);
    let model = SKMissingIndicator::<f64>::build()
        .build()
        .unwrap()
        .fit(fitted.view())
        .unwrap();
    let features = csr((2, 2), &[(0, 0, f64::NAN), (1, 0, f64::NAN), (0, 1, 7.0)]);
    match model.transform(features.view()).unwrap() {
        SKIndicatorOutput::Sparse(matrix) => {
            assert_eq!(matrix.shape(), (2, 1));
            assert_eq!(matrix.nnz(), 2, "exactly the missing cells are stored");
            assert_eq!(matrix.get(0, 0), Some(&1.0));
            assert_eq!(matrix.get(1, 0), Some(&1.0));
        }
        SKIndicatorOutput::Dense(matrix) => panic!("expected a csr mask, found {matrix:?}"),
    }
}

#[test]
fn sparse_output_requests_a_csr_mask_from_dense_input() {
    let model = SKMissingIndicator::<f64>::build()
        .features(SKMissingFeatureScope::All)
        .sparse_output(true)
        .build()
        .unwrap()
        .fit(&array![[1.0, 2.0], [3.0, 4.0]])
        .unwrap();
    match model
        .transform(&array![[f64::NAN, 1.0], [2.0, 3.0]])
        .unwrap()
    {
        SKIndicatorOutput::Sparse(matrix) => {
            assert_eq!(matrix.nnz(), 1);
            assert_eq!(matrix.get(0, 0), Some(&1.0));
        }
        SKIndicatorOutput::Dense(matrix) => panic!("expected a csr mask, found {matrix:?}"),
    }
}

#[test]
fn new_missingness_is_rejected_by_default() {
    let model = SKMissingIndicator::<f64>::build()
        .build()
        .unwrap()
        .fit(&array![[1.0, 2.0], [3.0, 4.0]])
        .unwrap();
    let error = model
        .transform(&array![[f64::NAN, 2.0]])
        .expect_err("a missing cell in a never-missing feature must fail");
    assert!(
        matches!(error, SKError::UnexpectedMissingness { column: 0 }),
        "expected the never-missing feature to be named, found {error:?}"
    );
}

#[test]
fn new_missingness_is_tolerated_on_request() {
    let model = SKMissingIndicator::<f64>::build()
        .error_on_new(false)
        .build()
        .unwrap()
        .fit(&array![[1.0, 2.0], [3.0, 4.0]])
        .unwrap();
    let mask = dense_mask(model.transform(&array![[f64::NAN, 2.0]]).unwrap());
    assert_eq!(mask.ncols(), 0, "no feature was missing at fit time");
}

#[test]
fn column_count_mismatch_is_rejected() {
    let model = SKMissingIndicator::<f64>::build()
        .build()
        .unwrap()
        .fit(&array![[1.0, 2.0], [3.0, 4.0]])
        .unwrap();
    let error = model
        .transform(&array![[1.0, 2.0, 3.0]])
        .expect_err("a column count mismatch must fail");
    assert!(matches!(error, SKError::ShapeMismatch { .. }));
}
