//! Companion tests for the validation surface and the dense-only contract
//! (spec `nearest-neighbor-imputation`).

use ndarray::{Array2, array};
use sciencekit_common::{SKBuilder, SKError, SKFeatureTransformer, SKUnsupervisedFit};
use sprs::TriMat;

use crate::knn::core_implementation::SKKNNImputer;

fn fitted(features: &Array2<f64>, k: usize) -> crate::knn::SKKNNImputerModel<f64> {
    SKKNNImputer::<f64>::build()
        .nearest_neighbors_count(k)
        .build()
        .unwrap()
        .fit(features.view())
        .unwrap()
}

#[test]
fn knn_validation_rejects_empty_fit_input() {
    let model = SKKNNImputer::<f64>::build()
        .nearest_neighbors_count(1)
        .build()
        .unwrap();
    let error = model
        .fit(&Array2::<f64>::zeros((0, 3)))
        .map(|_| ())
        .expect_err("an empty fit must fail");
    assert!(matches!(error, SKError::EmptyInput { operation: "fit" }));
    let error = model
        .fit(&Array2::<f64>::zeros((3, 0)))
        .map(|_| ())
        .expect_err("a zero-column fit must fail");
    assert!(matches!(error, SKError::EmptyInput { operation: "fit" }));
}

#[test]
fn knn_validation_rejects_empty_transform_input() {
    let model = fitted(&array![[1.0_f64, 2.0], [3.0, 4.0]], 2);
    let error = model
        .transform(&Array2::<f64>::zeros((0, 2)))
        .expect_err("an empty transform must fail");
    assert!(matches!(
        error,
        SKError::EmptyInput {
            operation: "transform"
        }
    ));
}

#[test]
fn knn_validation_rejects_a_zero_neighbor_count() {
    let error = SKKNNImputer::<f64>::build()
        .nearest_neighbors_count(0)
        .build()
        .map(|_| ())
        .expect_err("zero donors must fail");
    assert!(matches!(
        error,
        SKError::InvalidHyperparameter {
            name: "nearest_neighbors_count",
            ..
        }
    ));
}

#[test]
fn knn_validation_rejects_an_excessive_neighbor_count() {
    let error = SKKNNImputer::<f64>::build()
        .nearest_neighbors_count(3)
        .build()
        .unwrap()
        .fit(&array![[1.0_f64, 2.0], [3.0, 4.0]])
        .map(|_| ())
        .expect_err("more donors than training rows must fail");
    assert!(matches!(
        error,
        SKError::InvalidHyperparameter {
            name: "nearest_neighbors_count",
            ..
        }
    ));
}

#[test]
fn knn_validation_reports_a_feature_count_mismatch_with_both_dimensions() {
    let model = fitted(&array![[1.0_f64, 2.0], [3.0, 4.0]], 2);
    let error = model
        .transform(&array![[1.0_f64, 2.0, 3.0]])
        .expect_err("a feature-count mismatch must fail");
    assert!(matches!(
        error,
        SKError::ShapeMismatch {
            expected,
            found,
        } if expected == vec![1, 2] && found == vec![1, 3]
    ));
}

#[test]
fn knn_validation_rejects_nan_under_a_scalar_marker() {
    let error = SKKNNImputer::<f64>::build()
        .nearest_neighbors_count(2)
        .missing_values(-999.0)
        .build()
        .unwrap()
        .fit(&array![[1.0_f64, f64::NAN], [3.0, 4.0]])
        .map(|_| ())
        .expect_err("NaN is not the configured marker");
    assert!(matches!(error, SKError::NonFiniteValue { position: 1 }));
}

#[test]
fn knn_validation_rejects_an_infinity_under_the_nan_marker() {
    let error = SKKNNImputer::<f64>::build()
        .nearest_neighbors_count(2)
        .build()
        .unwrap()
        .fit(&array![[1.0_f64, f64::INFINITY], [3.0, 4.0]])
        .map(|_| ())
        .expect_err("an infinity is never a missing value");
    assert!(matches!(error, SKError::NonFiniteValue { position: 1 }));
}

#[test]
fn knn_validation_rejects_a_non_finite_value_at_transform() {
    let model = fitted(&array![[1.0_f64, 2.0], [3.0, 4.0]], 2);
    let error = model
        .transform(&array![[f64::INFINITY, 2.0]])
        .expect_err("an infinity must fail");
    assert!(matches!(error, SKError::NonFiniteValue { position: 0 }));
}

fn sparse_fixture() -> sprs::CsMat<f64> {
    let mut triplets = TriMat::new((2, 2));
    triplets.add_triplet(0, 0, 1.0);
    triplets.add_triplet(1, 1, 2.0);
    triplets.to_csr()
}

#[test]
fn knn_validation_rejects_sparse_fit_input() {
    let sparse = sparse_fixture();
    let error = SKKNNImputer::<f64>::build()
        .nearest_neighbors_count(1)
        .build()
        .unwrap()
        .fit(sparse.view())
        .map(|_| ())
        .expect_err("sparse fit must be rejected");
    match error {
        SKError::UnsupportedRepresentation {
            representation,
            suggestion,
        } => {
            assert_eq!(representation, "csr");
            assert!(suggestion.contains("SKSimpleImputer"));
        }
        other => panic!("expected UnsupportedRepresentation, found {other}"),
    }
}

#[test]
fn knn_validation_rejects_sparse_transform_input() {
    let model = fitted(&array![[1.0_f64, 2.0], [3.0, 4.0]], 2);
    let sparse = sparse_fixture();
    let error = model
        .transform(sparse.view())
        .expect_err("sparse transform must be rejected");
    assert!(matches!(
        error,
        SKError::UnsupportedRepresentation {
            representation: "csr",
            ..
        }
    ));
}
