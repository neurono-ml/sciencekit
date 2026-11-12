//! Companion tests for `SKTruncatedSVD` input, hyperparameter and shape
//! validation (`spec` `truncated-svd`).

use ndarray::{Array2, array};
use sciencekit_common::{SKBuilder, SKDataView, SKError, SKUnsupervisedFit};
use sprs::{CsMat, TriMat};

use super::{SKTruncatedSVD, SKTruncatedSVDAlgorithm};

struct FailingInput;

impl<'a> TryInto<SKDataView<'a, f64>> for FailingInput {
    type Error = SKError;
    fn try_into(self) -> Result<SKDataView<'a, f64>, Self::Error> {
        Err(SKError::Conversion("synthetic adapter failure".into()))
    }
}

fn sample_csr(rows: usize, columns: usize) -> CsMat<f64> {
    let mut triplet = TriMat::new((rows, columns));
    if rows > 0 && columns > 0 {
        triplet.add_triplet(0, 0, 2.5_f64);
    }
    if rows > 1 && columns > 1 {
        triplet.add_triplet(1, 1, -1.5_f64);
    }
    triplet.to_csr()
}

fn assert_empty(error: SKError, expected_operation: &'static str) {
    assert!(matches!(error, SKError::EmptyInput { operation } if operation == expected_operation));
}

fn assert_invalid(error: SKError, expected_name: &'static str) {
    assert!(matches!(error, SKError::InvalidHyperparameter { name, .. } if name == expected_name));
}

fn assert_mismatch(error: SKError, expected_shape: [usize; 2], found_shape: [usize; 2]) {
    assert!(
        matches!(error, SKError::ShapeMismatch { expected, found } if expected == expected_shape && found == found_shape)
    );
}

#[test]
fn truncated_validation_rejects_empty_inputs_and_propagates_conversion_failure() {
    let estimator = SKTruncatedSVD::<f64>::build()
        .number_of_components(1)
        .build()
        .unwrap();
    assert!(matches!(
        estimator.fit(FailingInput).unwrap_err(),
        SKError::Conversion(_)
    ));
    for (zero_rows, zero_columns) in [(0, 3), (3, 0)] {
        let empty_dense = Array2::<f64>::zeros((zero_rows, zero_columns));
        let empty_csr = sample_csr(zero_rows, zero_columns);
        assert_empty(estimator.fit(&empty_dense).unwrap_err(), "fit");
        assert_empty(estimator.fit(empty_csr.view()).unwrap_err(), "fit");
    }
    let zero_by_three = Array2::<f64>::zeros((0, 3));
    assert_empty(
        estimator.fit_transform(&zero_by_three).unwrap_err(),
        "fit_transform",
    );

    let valid = array![[1.0_f64, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];
    let fitted = estimator.fit(&valid).unwrap();
    let zero_by_one = Array2::<f64>::zeros((0, 1));
    assert_empty(fitted.transform(&zero_by_three).unwrap_err(), "transform");
    assert_empty(
        fitted.inverse_transform(&zero_by_one).unwrap_err(),
        "inverse_transform",
    );
}

#[test]
fn truncated_validation_rejects_zero_and_excessive_components_and_invalid_hyperparameters() {
    let build = SKTruncatedSVD::<f64>::build;
    assert_invalid(
        build().number_of_components(0).build().unwrap_err(),
        "number_of_components",
    );
    assert_invalid(
        build().number_of_iterations(0).build().unwrap_err(),
        "number_of_iterations",
    );
    for invalid_tolerance in [-1e-9_f64, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert_invalid(
            build().tolerance(invalid_tolerance).build().unwrap_err(),
            "tolerance",
        );
    }
    assert_invalid(
        build()
            .algorithm(SKTruncatedSVDAlgorithm::Arpack)
            .build()
            .unwrap_err(),
        "algorithm",
    );

    let wide = Array2::<f64>::from_elem((3, 5), 1.0);
    let tall = Array2::<f64>::from_elem((5, 3), 1.0);
    let four_components = build().number_of_components(4).build().unwrap();
    for invalid_fit in [
        four_components.fit(&wide),
        four_components.fit(&tall),
        four_components.fit(sample_csr(3, 5).view()),
        four_components.fit(sample_csr(5, 3).view()),
    ] {
        assert_invalid(invalid_fit.unwrap_err(), "number_of_components");
    }
}

#[test]
fn truncated_validation_rejects_feature_count_mismatch_non_finite_and_csc_views() {
    let training = array![
        [1.0_f64, 2.0, 3.0, 4.0],
        [5.0, 6.0, 7.0, 8.0],
        [9.0, 10.0, 11.0, 12.0]
    ];
    let estimator = SKTruncatedSVD::<f64>::build()
        .number_of_components(2)
        .build()
        .unwrap();
    let fitted = estimator.fit(&training).unwrap();

    for wrong_columns in [5usize, 0] {
        let wrong = Array2::<f64>::zeros((2, wrong_columns));
        assert_mismatch(
            fitted.transform(&wrong).unwrap_err(),
            [2, 4],
            [2, wrong_columns],
        );
    }
    assert_mismatch(
        fitted.transform(sample_csr(2, 5).view()).unwrap_err(),
        [2, 4],
        [2, 5],
    );
    for wrong_components in [3usize, 0] {
        let wrong = Array2::<f64>::zeros((2, wrong_components));
        assert_mismatch(
            fitted.inverse_transform(&wrong).unwrap_err(),
            [2, 2],
            [2, wrong_components],
        );
    }
    assert_mismatch(
        fitted
            .inverse_transform(sample_csr(2, 3).view())
            .unwrap_err(),
        [2, 2],
        [2, 3],
    );

    let dense_nan = array![[1.0_f64, 2.0, 3.0, 4.0], [5.0, 6.0, f64::NAN, 8.0]];
    assert!(matches!(
        estimator.fit(&dense_nan).unwrap_err(),
        SKError::NonFiniteValue { position: 6 }
    ));
    assert!(matches!(
        fitted.transform(&dense_nan).unwrap_err(),
        SKError::NonFiniteValue { position: 6 }
    ));
    let projected_nan = array![[1.0_f64, f64::NAN], [2.0, 3.0]];
    assert!(matches!(
        fitted.inverse_transform(&projected_nan).unwrap_err(),
        SKError::NonFiniteValue { position: 1 }
    ));
    let mut sparse_infinity = TriMat::new((3, 4));
    sparse_infinity.add_triplet(1, 2, f64::INFINITY);
    let csr_infinity = sparse_infinity.to_csr();
    assert!(matches!(
        estimator.fit(csr_infinity.view()).unwrap_err(),
        SKError::NonFiniteValue { position: 6 }
    ));

    for (csc_rows, csc_cols, is_inverse) in [(2, 4, false), (2, 2, true)] {
        let mut csc_triplet = TriMat::new((csc_rows, csc_cols));
        csc_triplet.add_triplet(0, 1, 1.0_f64);
        let csc = csc_triplet.to_csc();
        let err = if is_inverse {
            fitted.inverse_transform(csc.view()).unwrap_err()
        } else {
            fitted.transform(csc.view()).unwrap_err()
        };
        assert!(matches!(
            err,
            SKError::UnsupportedRepresentation {
                representation: "csc",
                ..
            }
        ));
    }
}
