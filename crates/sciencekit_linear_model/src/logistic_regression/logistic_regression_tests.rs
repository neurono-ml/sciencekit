//! Companion tests for `SKLogisticRegression` (spec `logistic-regression`).

use ndarray::{Array1, Array2, array};
use sciencekit_common::builders::SKBuilder;
use sciencekit_common::{SKClassifierPredictor, SKError, SKSupervisedFit};
use sprs::TriMat;

use super::SKLogisticRegression;
use crate::logistic::{SKClassWeight, SKLogisticPenalty, SKLogisticSolver, SKMultiClass};

fn binary_dataset() -> (Array2<f64>, Array1<i64>) {
    let features = array![
        [-2.0, -1.0],
        [-1.0, -2.0],
        [-1.0, -1.0],
        [1.0, 1.0],
        [2.0, 1.0],
        [1.0, 2.0]
    ];
    (features, array![0, 0, 0, 1, 1, 1])
}

fn to_csr(dense: &Array2<f64>) -> sprs::CsMat<f64> {
    let mut triplets = TriMat::new((dense.nrows(), dense.ncols()));
    dense.indexed_iter().for_each(|((row, column), &value)| {
        if value != 0.0 {
            triplets.add_triplet(row, column, value);
        }
    });
    triplets.to_csr()
}

#[test]
fn builder_defaults_are_documented() {
    let estimator = SKLogisticRegression::<f64>::build().build().unwrap();
    assert_eq!(estimator.penalty(), SKLogisticPenalty::L2);
    assert_eq!(estimator.solver(), SKLogisticSolver::Lbfgs);
    assert_eq!(estimator.multi_class(), SKMultiClass::Auto);
    assert!(estimator.include_intercept());
    assert_eq!(estimator.alpha(), 1.0);
    assert_eq!(estimator.c(), 1.0);
}

#[test]
fn c_and_alpha_are_reciprocal() {
    let estimator = SKLogisticRegression::<f64>::build().C(0.5).build().unwrap();
    assert_eq!(estimator.alpha(), 2.0);
    let estimator = SKLogisticRegression::<f64>::build()
        .C(0.5)
        .alpha(4.0)
        .build()
        .unwrap();
    assert_eq!(estimator.alpha(), 4.0);
}

#[test]
fn binary_fit_predicts_labels_and_probabilities() {
    let (features, labels) = binary_dataset();
    let model = SKLogisticRegression::<f64>::build()
        .build()
        .unwrap()
        .fit(features.view(), labels.view())
        .unwrap();
    let predicted = model.predict_labels(features.view()).unwrap();
    assert_eq!(predicted.to_vec(), labels.to_vec());
    let probabilities = model.predict_probabilities(features.view()).unwrap();
    assert_eq!(probabilities.ncols(), 2);
    for row in probabilities.rows() {
        let total: f64 = row.sum();
        assert!((total - 1.0).abs() < 1e-9);
    }
}

#[test]
fn nominal_labels_round_trip() {
    let (features, _) = binary_dataset();
    let labels = ["no", "no", "no", "yes", "yes", "yes"];
    let model = SKLogisticRegression::<f64>::build()
        .build()
        .unwrap()
        .fit(features.view(), &labels[..])
        .unwrap();
    assert_eq!(model.classes().number_of_classes(), 2);
    assert_eq!(model.classes().label_of(0), Some("no"));
    assert_eq!(model.classes().label_of(1), Some("yes"));
}

#[test]
fn multinomial_fits_three_classes() {
    let features = array![
        [-3.0, 0.0],
        [-2.0, 0.0],
        [0.0, 3.0],
        [0.0, 2.0],
        [3.0, 0.0],
        [2.0, 0.0]
    ];
    let labels = array![0_i64, 0, 1, 1, 2, 2];
    let model = SKLogisticRegression::<f64>::build()
        .build()
        .unwrap()
        .fit(features.view(), labels.view())
        .unwrap();
    assert_eq!(model.coefficients().nrows(), 3);
    let predicted = model.predict_labels(features.view()).unwrap();
    assert_eq!(predicted.to_vec(), labels.to_vec());
    let probabilities = model.predict_probabilities(features.view()).unwrap();
    assert_eq!(probabilities.ncols(), 3);
    for row in probabilities.rows() {
        let total: f64 = row.sum();
        assert!((total - 1.0).abs() < 1e-9);
    }
}

#[test]
fn liblinear_with_multinomial_is_rejected_at_fit() {
    let three = array![0_i64, 0, 1, 1, 2, 2];
    let features3 = array![
        [-3.0, 0.0],
        [-2.0, 0.0],
        [0.0, 3.0],
        [0.0, 2.0],
        [3.0, 0.0],
        [2.0, 0.0]
    ];
    let error = SKLogisticRegression::<f64>::build()
        .solver(SKLogisticSolver::Liblinear)
        .multi_class(SKMultiClass::Multinomial)
        .build()
        .unwrap()
        .fit(features3.view(), three.view())
        .unwrap_err();
    assert!(matches!(error, SKError::InvalidHyperparameter { name, .. } if name == "multi_class"));
}

#[test]
fn balanced_class_weight_fits_an_imbalanced_design() {
    let features = array![[-2.0], [-1.0], [0.5], [1.0], [2.0], [3.0]];
    let labels = array![0_i64, 0, 0, 0, 1, 1];
    let model = SKLogisticRegression::<f64>::build()
        .class_weight(SKClassWeight::Balanced)
        .build()
        .unwrap()
        .fit(features.view(), labels.view())
        .unwrap();
    assert_eq!(model.coefficients().nrows(), 1);
}

#[test]
fn warm_start_reaches_the_same_optimum() {
    let (features, labels) = binary_dataset();
    let cold = SKLogisticRegression::<f64>::build()
        .build()
        .unwrap()
        .fit(features.view(), labels.view())
        .unwrap();
    let warm = SKLogisticRegression::<f64>::build()
        .warm_start_from(&cold)
        .build()
        .unwrap()
        .fit(features.view(), labels.view())
        .unwrap();
    for (cold_value, warm_value) in cold.coefficients().iter().zip(warm.coefficients()) {
        assert!((cold_value - warm_value).abs() < 1e-6);
    }
}

#[test]
fn sparse_and_dense_agree() {
    let (features, labels) = binary_dataset();
    let sparse = to_csr(&features);
    let dense_model = SKLogisticRegression::<f64>::build()
        .build()
        .unwrap()
        .fit(features.view(), labels.view())
        .unwrap();
    let sparse_model = SKLogisticRegression::<f64>::build()
        .build()
        .unwrap()
        .fit(sparse.view(), labels.view())
        .unwrap();
    for (dense_value, sparse_value) in dense_model
        .coefficients()
        .iter()
        .zip(sparse_model.coefficients())
    {
        assert!((dense_value - sparse_value).abs() < 1e-6);
    }
}

#[test]
fn partial_fit_equals_the_dense_fit() {
    let features = array![
        [-2.0, -1.0],
        [-1.0, -2.0],
        [-1.0, -1.0],
        [1.0, 1.0],
        [2.0, 1.0],
        [1.0, 2.0]
    ];
    let labels = array![0_i64, 0, 0, 1, 1, 1];
    // Each batch carries both classes, so the first fit is a valid binary fit.
    let first = array![[-2.0, -1.0], [-1.0, -2.0], [1.0, 1.0], [2.0, 1.0]];
    let first_labels = array![0_i64, 0, 1, 1];
    let second = array![[-1.0, -1.0], [1.0, 2.0]];
    let second_labels = array![0_i64, 1];
    let mut model = SKLogisticRegression::<f64>::build()
        .build()
        .unwrap()
        .fit(first.view(), first_labels.view())
        .unwrap();
    model
        .partial_fit(second.view(), second_labels.view())
        .unwrap();
    let dense = SKLogisticRegression::<f64>::build()
        .build()
        .unwrap()
        .fit(features.view(), labels.view())
        .unwrap();
    for (partial, dense_value) in model.coefficients().iter().zip(dense.coefficients()) {
        assert!((partial - dense_value).abs() < 1e-6);
    }
}

#[test]
fn continuous_targets_and_single_class_are_rejected() {
    let features = array![[1.0], [2.0]];
    let continuous = array![0.1_f64, 0.9];
    assert!(
        SKLogisticRegression::<f64>::build()
            .build()
            .unwrap()
            .fit(features.view(), continuous.view())
            .is_err()
    );
    let single = array![0_i64, 0];
    assert!(
        SKLogisticRegression::<f64>::build()
            .build()
            .unwrap()
            .fit(features.view(), single.view())
            .is_err()
    );
}

#[test]
fn prediction_shape_mismatch_is_rejected() {
    let (features, labels) = binary_dataset();
    let model = SKLogisticRegression::<f64>::build()
        .build()
        .unwrap()
        .fit(features.view(), labels.view())
        .unwrap();
    let wrong = array![[1.0, 2.0, 3.0]];
    assert!(matches!(
        model.predict_labels(wrong.view()),
        Err(SKError::ShapeMismatch { .. })
    ));
}

#[test]
fn f32_pipelines_fit_and_predict() {
    let features = array![[-2.0_f32, -1.0], [1.0, 2.0], [2.0, 1.0]];
    let labels = array![0_i64, 1, 1];
    let model = SKLogisticRegression::<f32>::build()
        .build()
        .unwrap()
        .fit(features.view(), labels.view())
        .unwrap();
    let predicted = model.predict_labels(features.view()).unwrap();
    assert_eq!(predicted.len(), 3);
}

#[test]
fn a_shared_model_serves_concurrent_predictions() {
    let (features, labels) = binary_dataset();
    let model = std::sync::Arc::new(
        SKLogisticRegression::<f64>::build()
            .build()
            .unwrap()
            .fit(features.view(), labels.view())
            .unwrap(),
    );
    let handles: Vec<_> = (0..4)
        .map(|_| {
            let model = model.clone();
            let features = features.clone();
            std::thread::spawn(move || model.predict_labels(features.view()).unwrap())
        })
        .collect();
    let expected = model.predict_labels(features.view()).unwrap();
    for handle in handles {
        assert_eq!(handle.join().unwrap().to_vec(), expected.to_vec());
    }
}

#[test]
fn growing_the_penalty_shrinks_the_coefficients() {
    let (features, labels) = binary_dataset();
    let small = SKLogisticRegression::<f64>::build()
        .alpha(0.1)
        .build()
        .unwrap()
        .fit(features.view(), labels.view())
        .unwrap();
    let large = SKLogisticRegression::<f64>::build()
        .alpha(5.0)
        .build()
        .unwrap()
        .fit(features.view(), labels.view())
        .unwrap();
    let norm = |model: &crate::logistic_regression::SKLogisticRegressionModel<f64>| {
        model
            .coefficients()
            .iter()
            .map(|value| value.abs())
            .fold(0.0, f64::max)
    };
    assert!(norm(&large) < norm(&small));
}

#[test]
fn a_large_c_approaches_the_unregularized_fit() {
    let (features, labels) = binary_dataset();
    let unregularized = SKLogisticRegression::<f64>::build()
        .penalty(SKLogisticPenalty::None)
        .build()
        .unwrap()
        .fit(features.view(), labels.view())
        .unwrap();
    let large_c = SKLogisticRegression::<f64>::build()
        .C(1e8)
        .build()
        .unwrap()
        .fit(features.view(), labels.view())
        .unwrap();
    for (a, b) in unregularized
        .coefficients()
        .iter()
        .zip(large_c.coefficients())
    {
        assert!((a - b).abs() < 1e-3, "{a} vs {b}");
    }
}

#[test]
fn the_intercept_is_not_shrunk_by_the_penalty() {
    // A design whose best separator needs a large intercept.
    let features = array![[-2.0], [-1.0], [0.0], [1.0], [2.0]];
    let labels = array![0_i64, 0, 1, 1, 1];
    let small = SKLogisticRegression::<f64>::build()
        .alpha(0.01)
        .build()
        .unwrap()
        .fit(features.view(), labels.view())
        .unwrap();
    let large = SKLogisticRegression::<f64>::build()
        .alpha(10.0)
        .build()
        .unwrap()
        .fit(features.view(), labels.view())
        .unwrap();
    // The slope shrinks, but the intercept shrinks *less* than the slope — the
    // signature of an unpenalized intercept.
    let small_slope = small.coefficients()[[0, 0]].abs();
    let large_slope = large.coefficients()[[0, 0]].abs();
    let small_intercept = small.intercepts()[0].abs();
    let large_intercept = large.intercepts()[0].abs();
    assert!(large_slope < small_slope);
    assert!(large_intercept / small_intercept > large_slope / small_slope);
}

#[test]
fn the_decision_function_has_one_column_per_head_and_orders_the_classes() {
    let features = array![
        [-3.0, 0.0],
        [-2.0, 0.0],
        [0.0, 3.0],
        [0.0, 2.0],
        [3.0, 0.0],
        [2.0, 0.0]
    ];
    let labels = array![0_i64, 0, 1, 1, 2, 2];
    let model = SKLogisticRegression::<f64>::build()
        .build()
        .unwrap()
        .fit(features.view(), labels.view())
        .unwrap();
    let scores = model.decision_function(features.view()).unwrap();
    assert_eq!(scores.ncols(), 3);
    let probabilities = model.predict_probabilities(features.view()).unwrap();
    for row in 0..scores.nrows() {
        let best_score = scores
            .row(row)
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;
        let best_probability = probabilities
            .row(row)
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;
        assert_eq!(best_score, best_probability);
    }
}

#[test]
fn warm_start_without_a_source_equals_the_cold_fit() {
    let (features, labels) = binary_dataset();
    let cold = SKLogisticRegression::<f64>::build()
        .build()
        .unwrap()
        .fit(features.view(), labels.view())
        .unwrap();
    let warm = SKLogisticRegression::<f64>::build()
        .warm_start(true)
        .build()
        .unwrap()
        .fit(features.view(), labels.view())
        .unwrap();
    for (a, b) in cold.coefficients().iter().zip(warm.coefficients()) {
        assert!((a - b).abs() < 1e-9);
    }
}

#[test]
fn every_supported_solver_penalty_pair_converges() {
    // An XOR-like, deliberately non-separable design, so the unregularised MLE
    // is finite.
    let features = array![
        [-2.0, -2.0],
        [-2.0, 2.0],
        [2.0, -2.0],
        [2.0, 2.0],
        [-1.0, -1.0],
        [-1.0, 1.0]
    ];
    let labels = array![0_i64, 1, 1, 0, 0, 1];
    let pairs = [
        (SKLogisticSolver::Lbfgs, SKLogisticPenalty::L2),
        (SKLogisticSolver::Lbfgs, SKLogisticPenalty::None),
        (SKLogisticSolver::Liblinear, SKLogisticPenalty::L1),
        (SKLogisticSolver::Liblinear, SKLogisticPenalty::L2),
        (SKLogisticSolver::Saga, SKLogisticPenalty::L1),
        (SKLogisticSolver::Saga, SKLogisticPenalty::L2),
        (SKLogisticSolver::Saga, SKLogisticPenalty::ElasticNet),
        (SKLogisticSolver::Saga, SKLogisticPenalty::None),
    ];
    for (solver, penalty) in pairs {
        let model = SKLogisticRegression::<f64>::build()
            .solver(solver)
            .penalty(penalty)
            .alpha(0.5)
            .maximum_number_of_iterations(10_000)
            .tolerance(1e-6)
            .build()
            .unwrap()
            .fit(features.view(), labels.view())
            .unwrap_or_else(|error| panic!("{solver:?}/{penalty:?}: {error}"));
        assert!(
            model.coefficients().iter().all(|value| value.is_finite()),
            "{solver:?}/{penalty:?} produced a non-finite coefficient"
        );
    }
}

#[test]
fn an_empty_design_is_rejected() {
    let empty = Array2::<f64>::zeros((0, 2));
    let labels = Array1::<i64>::zeros(0);
    let result = SKLogisticRegression::<f64>::build()
        .build()
        .unwrap()
        .fit(empty.view(), labels.view());
    assert!(matches!(result, Err(SKError::ShapeMismatch { .. })));
}

#[test]
fn configuration_flows_through_the_model() {
    let (features, labels) = binary_dataset();
    let model = SKLogisticRegression::<f64>::build()
        .include_intercept(false)
        .backend(sciencekit_common::SKBackendKind::MatrixMultiply)
        .build()
        .unwrap()
        .fit(features.view(), labels.view())
        .unwrap();
    assert!(!model.include_intercept());
    assert_eq!(
        model.backend(),
        sciencekit_common::SKBackendKind::MatrixMultiply
    );
    assert!(model.intercepts().iter().all(|value| *value == 0.0));
}
