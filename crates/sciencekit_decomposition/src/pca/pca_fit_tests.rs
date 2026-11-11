//! Companion tests for `SKUnsupervisedFit` on `SKPCA` and `SKPCAModel` accessors.

use ndarray::{Array2, Axis, array};
use sciencekit_common::{SKBackendKind, SKBuilder, SKExecutionMode, SKUnsupervisedFit};

use super::test_tracing::PcaCaptureGuard;
use super::{SKPCA, SKPCASolver};

#[test]
fn pca_fit_populates_model_state_with_sorted_variance_and_zero_mean_residuals() {
    let data = array![
        [2.0_f64, 4.0, 1.0],
        [4.0, 8.0, 1.5],
        [6.0, 12.0, 0.5],
        [8.0, 16.0, 1.2],
        [10.0, 20.0, 0.8]
    ];
    let model = SKPCA::<f64>::build().build().unwrap().fit(&data).unwrap();

    assert_eq!(model.number_of_components(), 3);
    assert_eq!(model.number_of_features(), 3);
    assert_eq!(model.components().dim(), (3, 3));
    assert_eq!(model.explained_variance().len(), 3);
    assert_eq!(model.explained_variance_ratio().len(), 3);
    assert_eq!(model.singular_values().len(), 3);
    assert_eq!(model.mean().len(), 3);
    assert_eq!(model.resolved_solver(), SKPCASolver::Full);
    assert!(!model.whiten());
    assert_eq!(model.backend(), SKBackendKind::Faer);
    assert_eq!(model.execution_mode(), SKExecutionMode::Automatic);

    for window in model.explained_variance().as_slice().unwrap().windows(2) {
        assert!(window[0] >= window[1]);
    }
    let ratio_sum: f64 = model.explained_variance_ratio().iter().sum();
    assert!((ratio_sum - 1.0).abs() < 1e-12);

    let centered = &data - &model.mean().insert_axis(Axis(0));
    let column_means = centered.mean_axis(Axis(0)).unwrap();
    for &value in column_means.iter() {
        assert!(value.abs() < 1e-12);
    }
}

#[test]
fn pca_fit_transfers_configured_backend_and_records_fit_span() {
    let data = array![[1.0_f64, 2.0], [3.0, 5.0], [5.0, 8.0], [7.0, 11.0]];
    let capture = PcaCaptureGuard::begin();
    let model = SKPCA::<f64>::build()
        .number_of_components(1)
        .backend(SKBackendKind::NdArrayLinalg)
        .execution_mode(SKExecutionMode::InProcessSynchronous)
        .build()
        .unwrap()
        .fit(&data)
        .unwrap();

    assert_eq!(model.backend(), SKBackendKind::NdArrayLinalg);
    assert_eq!(
        model.execution_mode(),
        SKExecutionMode::InProcessSynchronous
    );

    let spans = capture.spans();
    let fit_span = spans
        .iter()
        .find(|(name, fields)| {
            name == "sciencekit.operation"
                && fields.get("operation").map(String::as_str) == Some("fit")
                && fields.get("backend").map(String::as_str) == Some("ndarray-linalg")
        })
        .expect("fit must record the configured backend in sciencekit.operation");
    assert_eq!(
        fit_span.1.get("mode").map(String::as_str),
        Some("InProcessSynchronous")
    );
}

#[test]
fn pca_fit_agrees_across_all_solvers_on_low_rank_design() {
    let data = Array2::from_shape_fn((45, 6), |(row, column)| {
        let row_scalar = row as f64;
        let column_scalar = (column + 1) as f64;
        3.0 * (0.19 * row_scalar).sin() * column_scalar
            + 1.4 * (0.37 * row_scalar).cos() * (7.0 - column_scalar)
            + 5.0 * column_scalar
    });
    let auto_small = SKPCA::<f64>::build()
        .number_of_components(2)
        .build()
        .unwrap()
        .fit(&data)
        .unwrap();
    assert_eq!(auto_small.resolved_solver(), SKPCASolver::Full);

    let full = SKPCA::<f64>::build()
        .number_of_components(2)
        .svd_solver(SKPCASolver::Full)
        .build()
        .unwrap()
        .fit(&data)
        .unwrap();
    let randomized = SKPCA::<f64>::build()
        .number_of_components(2)
        .svd_solver(SKPCASolver::Randomized)
        .random_seed(7)
        .build()
        .unwrap()
        .fit(&data)
        .unwrap();
    let covariance = SKPCA::<f64>::build()
        .number_of_components(2)
        .svd_solver(SKPCASolver::CovarianceEigh)
        .build()
        .unwrap()
        .fit(&data)
        .unwrap();

    assert_eq!(full.resolved_solver(), SKPCASolver::Full);
    assert_eq!(randomized.resolved_solver(), SKPCASolver::Randomized);
    assert_eq!(covariance.resolved_solver(), SKPCASolver::CovarianceEigh);

    for other in [&randomized, &covariance] {
        for (left, right) in full.components().iter().zip(other.components().iter()) {
            assert!((left - right).abs() < 1e-6);
        }
        for (left, right) in full
            .explained_variance()
            .iter()
            .zip(other.explained_variance().iter())
        {
            assert!((left - right).abs() < 1e-6);
        }
        for (left, right) in full
            .explained_variance_ratio()
            .iter()
            .zip(other.explained_variance_ratio().iter())
        {
            assert!((left - right).abs() < 1e-7);
        }
    }
}
