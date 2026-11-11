//! Companion tests for PCA representation policy, explicit densification and
//! input validation.

use ndarray::{Array2, array};
use sciencekit_common::{SKBuilder, SKError, SKUnsupervisedFit};
use sprs::{CsMat, TriMat};

use super::test_tracing::PcaCaptureGuard;
use super::{SKPCA, SKPCASolver};

fn csr_from_dense(dense: &Array2<f64>) -> CsMat<f64> {
    let mut triplet = TriMat::new(dense.dim());
    for ((row, column), &value) in dense.indexed_iter() {
        if value != 0.0 {
            triplet.add_triplet(row, column, value);
        }
    }
    triplet.to_csr()
}

#[test]
fn pca_validation_sparse_randomized_matches_dense_without_densifying() {
    let mut dense = Array2::<f64>::zeros((30, 8));
    for row in 0..30 {
        dense[(row, row % 8)] = (row as f64 + 1.0) * 1.25;
        dense[(row, (row * 3 + 1) % 8)] = -((row % 4) as f64 + 0.5) * 2.0;
    }
    let sparse = csr_from_dense(&dense);
    let estimator = SKPCA::<f64>::build()
        .number_of_components(2)
        .svd_solver(SKPCASolver::Randomized)
        .random_seed(11)
        .build()
        .unwrap();

    let capture = PcaCaptureGuard::begin();
    let sparse_model = estimator.fit(sparse.view()).unwrap();
    assert!(capture.warning_events().is_empty());

    let dense_model = estimator.fit(&dense).unwrap();
    for (left, right) in sparse_model
        .components()
        .iter()
        .zip(dense_model.components().iter())
    {
        assert!((left - right).abs() < 1e-10);
    }
    for (left, right) in sparse_model
        .explained_variance_ratio()
        .iter()
        .zip(dense_model.explained_variance_ratio().iter())
    {
        assert!((left - right).abs() < 1e-10);
    }
    let sparse_projected = sparse_model.transform(sparse.view()).unwrap();
    let dense_projected = dense_model.transform(&dense).unwrap();
    for (left, right) in sparse_projected.iter().zip(dense_projected.iter()) {
        assert!((left - right).abs() < 1e-10);
    }
}

#[test]
fn pca_validation_full_and_covariance_reject_sparse_by_default_with_alternatives() {
    let sparse = csr_from_dense(&array![
        [0.0_f64, 3.0, 0.0],
        [2.0, 0.0, 5.0],
        [0.0, 1.0, 4.0]
    ]);
    for solver in [SKPCASolver::Full, SKPCASolver::CovarianceEigh] {
        let error = SKPCA::<f64>::build()
            .number_of_components(2)
            .svd_solver(solver)
            .build()
            .unwrap()
            .fit(sparse.view())
            .unwrap_err();
        match error {
            SKError::UnsupportedRepresentation {
                representation,
                suggestion,
            } => {
                assert_eq!(representation, "csr");
                assert!(suggestion.contains("Randomized"));
                assert!(suggestion.contains("SKTruncatedSVD"));
            }
            other => panic!("expected UnsupportedRepresentation, found {other:?}"),
        }
    }
}

#[test]
fn pca_validation_allow_densification_warns_with_memory_cost_and_matches_dense() {
    let dense = Array2::from_shape_fn((10, 5), |(row, column)| {
        if (row + column) % 3 == 0 {
            (row as f64 + 1.0) * (column as f64 - 2.0)
        } else {
            0.0
        }
    });
    let sparse = csr_from_dense(&dense);
    for solver in [SKPCASolver::Full, SKPCASolver::CovarianceEigh] {
        let capture = PcaCaptureGuard::begin();
        let sparse_model = SKPCA::<f64>::build()
            .number_of_components(2)
            .svd_solver(solver)
            .allow_densification(true)
            .build()
            .unwrap()
            .fit(sparse.view())
            .unwrap();
        let warnings = capture.warning_events();
        assert_eq!(warnings.len(), 1);
        assert_eq!(
            warnings[0].get("estimated_dense_bytes").map(String::as_str),
            Some("400")
        );
        let dense_model = SKPCA::<f64>::build()
            .number_of_components(2)
            .svd_solver(solver)
            .build()
            .unwrap()
            .fit(&dense)
            .unwrap();
        for (left, right) in sparse_model
            .components()
            .iter()
            .zip(dense_model.components().iter())
        {
            assert!((left - right).abs() < 1e-10);
        }
    }
}

#[test]
fn pca_validation_rejects_empty_excessive_non_finite_and_mismatched_shapes() {
    let estimator = SKPCA::<f64>::build().build().unwrap();
    assert!(matches!(
        estimator.fit(&Array2::<f64>::zeros((0, 3))).unwrap_err(),
        SKError::EmptyInput { operation: "fit" }
    ));
    assert!(matches!(
        estimator.fit(&Array2::<f64>::zeros((3, 0))).unwrap_err(),
        SKError::EmptyInput { operation: "fit" }
    ));
    assert!(matches!(
        estimator
            .fit_transform(&Array2::<f64>::zeros((0, 3)))
            .unwrap_err(),
        SKError::EmptyInput {
            operation: "fit_transform"
        }
    ));

    let small = array![
        [1.0_f64, 2.0, 3.0, 4.0],
        [5.0, 6.0, 7.0, 8.0],
        [9.0, 10.0, 11.0, 12.0]
    ];
    let excessive = SKPCA::<f64>::build()
        .number_of_components(4)
        .build()
        .unwrap()
        .fit(&small)
        .unwrap_err();
    assert!(matches!(
        excessive,
        SKError::InvalidHyperparameter {
            name: "number_of_components",
            ..
        }
    ));
    let with_nan = array![[1.0_f64, 2.0], [3.0, f64::NAN], [5.0, 6.0]];
    assert!(matches!(
        estimator.fit(&with_nan).unwrap_err(),
        SKError::NonFiniteValue { position: 3 }
    ));

    let fitted = SKPCA::<f64>::build()
        .number_of_components(2)
        .build()
        .unwrap()
        .fit(&small)
        .unwrap();
    assert!(matches!(
        fitted.transform(&Array2::<f64>::zeros((2, 5))).unwrap_err(),
        SKError::ShapeMismatch { expected, found } if expected == vec![2, 4] && found == vec![2, 5]
    ));
    assert!(matches!(
        fitted
            .inverse_transform(&Array2::<f64>::zeros((2, 3)))
            .unwrap_err(),
        SKError::ShapeMismatch { expected, found } if expected == vec![2, 2] && found == vec![2, 3]
    ));
    assert!(matches!(
        fitted.transform(&Array2::<f64>::zeros((0, 4))).unwrap_err(),
        SKError::EmptyInput {
            operation: "transform"
        }
    ));
}
