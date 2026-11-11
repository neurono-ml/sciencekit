//! Companion tests for randomized-SVD fitting and `SKTruncatedSVDModel` state.

use ndarray::{Array2, array};
use sciencekit_common::{SKBackendKind, SKBuilderState, SKDataView};

use super::config::{SKTruncatedSVDAlgorithm, SKTruncatedSVDConfiguration};
use super::fit::sk_fit_with_projection;

fn default_test_configuration(number_of_components: usize) -> SKTruncatedSVDConfiguration<f64> {
    SKTruncatedSVDConfiguration {
        number_of_components,
        algorithm: SKTruncatedSVDAlgorithm::Randomized,
        number_of_iterations: 5,
        tolerance: 0.0,
        random_seed: 0,
        backend: SKBackendKind::Faer,
        builder_state: SKBuilderState::new(),
    }
}

#[test]
fn truncated_fit_reconstructs_rank_one_matrix_without_centering() {
    // Rank-1 matrix u * v^T with non-zero feature means:
    // u = [1, 2, 3, 4]^T (||u|| = sqrt(30)), v = [0.6, 0.8, 0.0]^T (||v|| = 1).
    // Exact singular value sigma_1 = sqrt(30).
    let weights = array![1.0_f64, 2.0, 3.0, 4.0];
    let direction = array![0.6_f64, 0.8, 0.0];
    let data = Array2::from_shape_fn((4, 3), |(row, column)| weights[row] * direction[column]);

    let configuration = default_test_configuration(1);
    let (model, projected) =
        sk_fit_with_projection(&configuration, SKDataView::Dense(data.view()), "fit").unwrap();

    assert_eq!(model.number_of_components(), 1);
    assert_eq!(model.number_of_features(), 3);
    assert!((model.singular_values()[0] - 30.0_f64.sqrt()).abs() < 1e-10);

    // Uncentered reconstruction: Z * components reproduces the non-zero means directly.
    let reconstructed = projected.dot(&model.components());
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

    let configuration = default_test_configuration(2);
    let (model, projected) =
        sk_fit_with_projection(&configuration, SKDataView::Dense(data.view()), "fit").unwrap();

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

    let reconstructed = projected.dot(&model.components());
    for (actual, expected) in reconstructed.iter().zip(data.iter()) {
        assert!((actual - expected).abs() < 1e-10);
    }
}

#[test]
fn truncated_fit_completes_orthonormal_components_on_constant_and_rank_deficient_data() {
    // Non-zero constant 6 x 3 matrix: rank 1 uncentered, zero column variance.
    let constant = Array2::from_elem((6, 3), 4.0_f64);
    let configuration = default_test_configuration(2);
    let (model, projected) =
        sk_fit_with_projection(&configuration, SKDataView::Dense(constant.view()), "fit").unwrap();

    let gram = model.components().dot(&model.components().t());
    for ((row, column), &entry) in gram.indexed_iter() {
        let expected = if row == column { 1.0 } else { 0.0 };
        assert!((entry - expected).abs() < 1e-12);
    }
    assert!((model.singular_values()[0] - (18.0_f64 * 16.0).sqrt()).abs() < 1e-10);
    assert!(model.singular_values()[1] < 1e-6);
    assert!(model.explained_variance().iter().all(|&v| v.abs() < 1e-12));
    assert!(model.explained_variance_ratio().iter().all(|&r| r == 0.0));

    let reconstructed = projected.dot(&model.components());
    for (actual, expected) in reconstructed.iter().zip(constant.iter()) {
        assert!((actual - expected).abs() < 1e-10);
    }
}
