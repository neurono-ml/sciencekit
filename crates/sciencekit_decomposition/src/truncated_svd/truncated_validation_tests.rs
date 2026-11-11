//! Companion tests for `SKTruncatedSVD` input, hyperparameter and shape
//! validation (`spec` `truncated-svd`).

use ndarray::{Array2, array};
use sciencekit_common::{SKBuilder, SKError, SKUnsupervisedFit};
use sprs::{CsMat, TriMat};

use super::{SKTruncatedSVD, SKTruncatedSVDAlgorithm};

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

fn assert_empty_input(error: SKError, expected_operation: &'static str) {
    match error {
        SKError::EmptyInput { operation } => assert_eq!(operation, expected_operation),
        other => panic!("expected EmptyInput({expected_operation}), found {other:?}"),
    }
}

fn assert_invalid_parameter(error: SKError, expected_name: &'static str) {
    match error {
        SKError::InvalidHyperparameter { name, .. } => assert_eq!(name, expected_name),
        other => panic!("expected InvalidHyperparameter({expected_name}), found {other:?}"),
    }
}

fn assert_shape_mismatch(error: SKError, expected_shape: [usize; 2], found_shape: [usize; 2]) {
    match error {
        SKError::ShapeMismatch { expected, found } => {
            assert_eq!(expected, expected_shape);
            assert_eq!(found, found_shape);
        }
        other => panic!("expected ShapeMismatch, found {other:?}"),
    }
}

#[test]
fn truncated_validation_rejects_empty_dense_and_sparse_inputs() {
    let estimator = SKTruncatedSVD::<f64>::build()
        .number_of_components(1)
        .build()
        .unwrap();

    assert_empty_input(
        estimator.fit(&Array2::<f64>::zeros((0, 3))).unwrap_err(),
        "fit",
    );
    assert_empty_input(
        estimator.fit(&Array2::<f64>::zeros((3, 0))).unwrap_err(),
        "fit",
    );
    assert_empty_input(estimator.fit(sample_csr(0, 3).view()).unwrap_err(), "fit");
    assert_empty_input(estimator.fit(sample_csr(3, 0).view()).unwrap_err(), "fit");
    assert_empty_input(
        estimator
            .fit_transform(&Array2::<f64>::zeros((0, 3)))
            .unwrap_err(),
        "fit_transform",
    );

    let valid = array![[1.0_f64, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];
    let fitted = estimator.fit(&valid).unwrap();
    assert_empty_input(
        fitted.transform(&Array2::<f64>::zeros((0, 3))).unwrap_err(),
        "transform",
    );
    assert_empty_input(
        fitted
            .inverse_transform(&Array2::<f64>::zeros((0, 1)))
            .unwrap_err(),
        "inverse_transform",
    );
}

#[test]
fn truncated_validation_rejects_zero_and_excessive_components_and_invalid_hyperparameters() {
    assert_invalid_parameter(
        SKTruncatedSVD::<f64>::build()
            .number_of_components(0)
            .build()
            .unwrap_err(),
        "number_of_components",
    );
    assert_invalid_parameter(
        SKTruncatedSVD::<f64>::build()
            .number_of_iterations(0)
            .build()
            .unwrap_err(),
        "number_of_iterations",
    );
    for invalid_tolerance in [-1e-9_f64, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert_invalid_parameter(
            SKTruncatedSVD::<f64>::build()
                .tolerance(invalid_tolerance)
                .build()
                .unwrap_err(),
            "tolerance",
        );
    }
    assert_invalid_parameter(
        SKTruncatedSVD::<f64>::build()
            .algorithm(SKTruncatedSVDAlgorithm::Arpack)
            .build()
            .unwrap_err(),
        "algorithm",
    );

    let wide = Array2::<f64>::from_elem((3, 5), 1.0);
    let tall = Array2::<f64>::from_elem((5, 3), 1.0);
    let four_components = SKTruncatedSVD::<f64>::build()
        .number_of_components(4)
        .build()
        .unwrap();
    for invalid_fit in [
        four_components.fit(&wide),
        four_components.fit(&tall),
        four_components.fit(sample_csr(3, 5).view()),
        four_components.fit(sample_csr(5, 3).view()),
    ] {
        assert_invalid_parameter(invalid_fit.unwrap_err(), "number_of_components");
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

    assert_shape_mismatch(
        fitted.transform(&Array2::<f64>::zeros((2, 5))).unwrap_err(),
        [2, 4],
        [2, 5],
    );
    assert_shape_mismatch(
        fitted.transform(&Array2::<f64>::zeros((2, 0))).unwrap_err(),
        [2, 4],
        [2, 0],
    );
    assert_shape_mismatch(
        fitted.transform(sample_csr(2, 5).view()).unwrap_err(),
        [2, 4],
        [2, 5],
    );
    assert_shape_mismatch(
        fitted
            .inverse_transform(&Array2::<f64>::zeros((2, 3)))
            .unwrap_err(),
        [2, 2],
        [2, 3],
    );
    assert_shape_mismatch(
        fitted
            .inverse_transform(&Array2::<f64>::zeros((2, 0)))
            .unwrap_err(),
        [2, 2],
        [2, 0],
    );
    assert_shape_mismatch(
        fitted
            .inverse_transform(sample_csr(2, 3).view())
            .unwrap_err(),
        [2, 2],
        [2, 3],
    );

    let dense_with_nan = array![[1.0_f64, 2.0, 3.0, 4.0], [5.0, 6.0, f64::NAN, 8.0]];
    assert!(matches!(
        estimator.fit(&dense_with_nan).unwrap_err(),
        SKError::NonFiniteValue { position: 6 }
    ));
    assert!(matches!(
        fitted.transform(&dense_with_nan).unwrap_err(),
        SKError::NonFiniteValue { position: 6 }
    ));

    let mut sparse_with_infinity = TriMat::new((3, 4));
    sparse_with_infinity.add_triplet(1, 2, f64::INFINITY);
    assert!(matches!(
        estimator
            .fit(sparse_with_infinity.to_csr().view())
            .unwrap_err(),
        SKError::NonFiniteValue { position: 6 }
    ));

    let mut csc_triplet = TriMat::new((2, 4));
    csc_triplet.add_triplet(0, 1, 1.0_f64);
    assert!(matches!(
        fitted.transform(csc_triplet.to_csc().view()).unwrap_err(),
        SKError::UnsupportedRepresentation {
            representation: "csc",
            ..
        }
    ));
}
