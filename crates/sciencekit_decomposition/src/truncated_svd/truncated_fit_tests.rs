//! Companion tests for `SKUnsupervisedFit` on `SKTruncatedSVD`, `fit_transform`
//! and `SKTruncatedSVDModel` state accessors.

use ndarray::{Array2, array};
use sciencekit_common::{SKBackendKind, SKBuilder, SKExecutionMode, SKUnsupervisedFit};

use super::{SKTruncatedSVD, SKTruncatedSVDAlgorithm};
use crate::pca::test_tracing::PcaCaptureGuard;

#[test]
fn truncated_fit_reconstructs_rank_one_matrix_without_centering() {
    // Rank-1 matrix u * v^T with non-zero feature means:
    // u = [1, 2, 3, 4]^T (||u|| = sqrt(30)), v = [0.6, 0.8, 0.0]^T (||v|| = 1).
    // Exact singular value sigma_1 = sqrt(30).
    let weights = array![1.0_f64, 2.0, 3.0, 4.0];
    let direction = array![0.6_f64, 0.8, 0.0];
    let data = Array2::from_shape_fn((4, 3), |(row, column)| weights[row] * direction[column]);

    let estimator = SKTruncatedSVD::<f64>::build()
        .number_of_components(1)
        .build()
        .unwrap();
    let model = estimator.fit(&data).unwrap();
    let projected = estimator.fit_transform(&data).unwrap();
    assert_eq!(projected, model.transform(&data).unwrap());

    assert_eq!(model.number_of_components(), 1);
    assert_eq!(model.number_of_features(), 3);
    assert_eq!(model.algorithm(), SKTruncatedSVDAlgorithm::Randomized);
    assert_eq!(model.backend(), SKBackendKind::Faer);
    assert_eq!(model.execution_mode(), SKExecutionMode::Automatic);
    assert!((model.singular_values()[0] - 30.0_f64.sqrt()).abs() < 1e-10);

    // Uncentered reconstruction: inverse_transform reproduces the non-zero means directly.
    let reconstructed = model.inverse_transform(&projected).unwrap();
    for (actual, expected) in reconstructed.iter().zip(data.iter()) {
        assert!((actual - expected).abs() < 1e-10);
    }

    // Variance of weights = Var([1, 2, 3, 4]) = 1.25; ||direction|| = 1 => ratio = 1.0.
    assert!((model.explained_variance()[0] - 1.25).abs() < 1e-10);
    assert!((model.explained_variance_ratio()[0] - 1.0).abs() < 1e-10);
    assert!(model.explained_variance_ratio().sum() <= 1.0);
}

#[test]
fn truncated_fit_reconstructs_rank_two_matrix_with_non_increasing_variance() {
    // Orthogonal feature directions v1 = [1, 0, 0, 0], v2 = [0, 0.6, 0.8, 0]
    // with non-zero means on both modes and strictly larger spread on mode 1.
    let mode_one = array![2.0_f64, 5.0, 8.0, 11.0, 14.0];
    let mode_two = array![1.0_f64, 2.0, 1.5, 2.5, 3.0];
    let direction_one = array![1.0_f64, 0.0, 0.0, 0.0];
    let direction_two = array![0.0_f64, 0.6, 0.8, 0.0];
    let data = Array2::from_shape_fn((5, 4), |(row, column)| {
        mode_one[row] * direction_one[column] + mode_two[row] * direction_two[column]
    });

    let model = SKTruncatedSVD::<f64>::build()
        .number_of_components(2)
        .build()
        .unwrap()
        .fit(&data)
        .unwrap();

    assert_eq!(model.components().dim(), (2, 4));
    let gram = model.components().dot(&model.components().t());
    for ((row, column), &entry) in gram.indexed_iter() {
        let expected = if row == column { 1.0 } else { 0.0 };
        assert!((entry - expected).abs() < 1e-10);
    }

    assert!(model.singular_values()[0] >= model.singular_values()[1]);
    assert!(model.explained_variance()[0] >= model.explained_variance()[1]);
    let ratio_sum: f64 = model.explained_variance_ratio().sum();
    assert!(ratio_sum <= 1.0);
    assert!((ratio_sum - 1.0).abs() < 1e-10);

    let projected = model.transform(&data).unwrap();
    let reconstructed = model.inverse_transform(&projected).unwrap();
    for (actual, expected) in reconstructed.iter().zip(data.iter()) {
        assert!((actual - expected).abs() < 1e-10);
    }
}

#[test]
fn truncated_fit_completes_orthonormal_components_on_constant_and_rank_deficient_data() {
    // Non-zero constant 6 x 3 matrix: rank 1 uncentered, zero column variance.
    let constant = Array2::from_elem((6, 3), 4.0_f64);
    let model = SKTruncatedSVD::<f64>::build()
        .number_of_components(2)
        .build()
        .unwrap()
        .fit(&constant)
        .unwrap();

    let gram = model.components().dot(&model.components().t());
    for ((row, column), &entry) in gram.indexed_iter() {
        let expected = if row == column { 1.0 } else { 0.0 };
        assert!((entry - expected).abs() < 1e-12);
    }
    assert!((model.singular_values()[0] - (18.0_f64 * 16.0).sqrt()).abs() < 1e-10);
    assert!(model.singular_values()[1] < 1e-6);
    assert!(
        model
            .explained_variance()
            .iter()
            .all(|&variance| variance.abs() < 1e-12)
    );
    assert!(
        model
            .explained_variance_ratio()
            .iter()
            .all(|&ratio| ratio == 0.0)
    );

    let projected = model.transform(&constant).unwrap();
    let reconstructed = model.inverse_transform(&projected).unwrap();
    for (actual, expected) in reconstructed.iter().zip(constant.iter()) {
        assert!((actual - expected).abs() < 1e-10);
    }
}

#[test]
fn truncated_fit_transfers_configured_backend_and_records_fit_and_fit_transform_spans() {
    let data = array![[1.0_f64, 2.0], [3.0, 5.0], [5.0, 8.0], [7.0, 11.0]];
    let estimator = SKTruncatedSVD::<f64>::build()
        .number_of_components(1)
        .backend(SKBackendKind::NdArrayLinalg)
        .execution_mode(SKExecutionMode::InProcessSynchronous)
        .build()
        .unwrap();

    let capture = PcaCaptureGuard::begin();
    let model = estimator.fit(&data).unwrap();
    let projected = estimator.fit_transform(&data).unwrap();
    let spans = capture.spans();
    drop(capture);

    assert_eq!(model.backend(), SKBackendKind::NdArrayLinalg);
    assert_eq!(
        model.execution_mode(),
        SKExecutionMode::InProcessSynchronous
    );
    assert_eq!(projected, model.transform(&data).unwrap());

    for expected_operation in ["fit", "fit_transform"] {
        let recorded = spans
            .iter()
            .find(|(name, fields)| {
                name == "sciencekit.operation"
                    && fields.get("operation").map(String::as_str) == Some(expected_operation)
                    && fields.get("backend").map(String::as_str) == Some("ndarray-linalg")
            })
            .unwrap_or_else(|| {
                panic!(
                    "{expected_operation} must record configured backend in sciencekit.operation"
                )
            });
        assert_eq!(
            recorded.1.get("mode").map(String::as_str),
            Some("InProcessSynchronous")
        );
    }

    let default_estimator = SKTruncatedSVD::<f64>::build()
        .number_of_components(1)
        .build()
        .unwrap();
    let default_capture = PcaCaptureGuard::begin();
    let default_model = default_estimator.fit(&data).unwrap();
    let _ = default_model.transform(&data).unwrap();
    let default_spans = default_capture.spans();
    for expected_operation in ["fit", "transform"] {
        assert!(default_spans.iter().any(|(name, fields)| {
            name == "sciencekit.operation"
                && fields.get("operation").map(String::as_str) == Some(expected_operation)
                && fields.get("backend").map(String::as_str) == Some("faer")
        }));
    }
}
