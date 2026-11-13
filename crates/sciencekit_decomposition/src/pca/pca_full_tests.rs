//! Companion tests for the full-SVD PCA solver.

use ndarray::array;
use sciencekit_common::{SKBackendKind, SKDataView};
use sciencekit_math::backend::sk_math_backend_for_kind;

use super::centering::{sk_center_dense, sk_feature_means, sk_total_centered_variance};
use super::solvers::sk_solve_full_svd;

#[test]
fn pca_full_recovers_diagonal_principal_axes_and_reconstructs_centered_data() {
    let data = array![
        [-3.0_f64, -3.0],
        [-1.0, -1.0],
        [1.0, 1.0],
        [3.0, 3.0],
        [-0.2, 0.2],
        [0.2, -0.2]
    ];
    let mean = sk_feature_means(SKDataView::Dense(data.view()), 1).unwrap();
    let centered = sk_center_dense(data.view(), mean.view());
    let total_variance =
        sk_total_centered_variance(SKDataView::Dense(data.view()), mean.view()).unwrap();
    let backend = sk_math_backend_for_kind::<f64>(SKBackendKind::Faer);

    let full = sk_solve_full_svd(backend.as_ref(), centered.view(), 2, total_variance).unwrap();
    let inverse_root_two = 1.0_f64 / 2.0_f64.sqrt();
    assert!((full.components[(0, 0)] - inverse_root_two).abs() < 1e-12);
    assert!((full.components[(0, 1)] - inverse_root_two).abs() < 1e-12);
    assert!(full.explained_variance[0] > full.explained_variance[1]);
    assert!(full.singular_values[0] > full.singular_values[1]);
    let degrees_of_freedom = (data.nrows() - 1) as f64;
    assert!(
        (full.explained_variance[0]
            - (full.singular_values[0] * full.singular_values[0]) / degrees_of_freedom)
            .abs()
            < 1e-12
    );

    let gram = full.components.dot(&full.components.t());
    assert!((gram[(0, 0)] - 1.0).abs() < 1e-12);
    assert!((gram[(1, 1)] - 1.0).abs() < 1e-12);
    assert!(gram[(0, 1)].abs() < 1e-12);

    let ratio_sum: f64 = full.explained_variance_ratio.iter().sum();
    assert!((ratio_sum - 1.0).abs() < 1e-12);

    let reconstructed = centered.dot(&full.components.t()).dot(&full.components);
    for (left, right) in reconstructed.iter().zip(centered.iter()) {
        assert!((left - right).abs() < 1e-11);
    }

    let truncated =
        sk_solve_full_svd(backend.as_ref(), centered.view(), 1, total_variance).unwrap();
    assert_eq!(truncated.components.dim(), (1, 2));
    assert!(
        (truncated.explained_variance_ratio[0] - full.explained_variance_ratio[0]).abs() < 1e-12
    );
    assert!(truncated.explained_variance_ratio[0] < 1.0);
}

#[test]
fn pca_full_handles_zero_variance_without_non_finite_ratios() {
    let constant = array![[5.0_f64, -2.0], [5.0, -2.0], [5.0, -2.0]];
    let mean = sk_feature_means(SKDataView::Dense(constant.view()), 1).unwrap();
    let centered = sk_center_dense(constant.view(), mean.view());
    let total_variance =
        sk_total_centered_variance(SKDataView::Dense(constant.view()), mean.view()).unwrap();
    let backend = sk_math_backend_for_kind::<f64>(SKBackendKind::Faer);

    let result = sk_solve_full_svd(backend.as_ref(), centered.view(), 2, total_variance).unwrap();
    assert!(result.explained_variance.iter().all(|&value| value == 0.0));
    assert!(
        result
            .explained_variance_ratio
            .iter()
            .all(|&value| value.is_finite() && value == 0.0)
    );
}
