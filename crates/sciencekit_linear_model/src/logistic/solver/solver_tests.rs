//! Companion tests for the logistic solvers (spec `logistic-solver`).

use ndarray::{Array1, Array2, array};

use crate::logistic::test_datasets::{
    assert_close_scalar, separable_binary_design_with_zero_column, three_class_design, to_csr,
};

use super::orchestration::sk_solve_logistic;
use super::solution::SKLogisticSolution;
use crate::logistic::config::{
    SKClassWeight, SKLogisticConfig, SKLogisticPenalty, SKLogisticSolver, SKMultiClass,
};
use crate::logistic::design::{SKDenseDesign, SKSparseDesign};
use crate::logistic::resolution::SKResolvedMultiClass;

fn config(
    solver: SKLogisticSolver,
    penalty: SKLogisticPenalty,
    alpha: f64,
    include_intercept: bool,
) -> SKLogisticConfig<f64> {
    SKLogisticConfig {
        alpha,
        tolerance: 1e-8,
        maximum_number_of_iterations: 5_000,
        penalty,
        l1_ratio: 1.0,
        solver,
        multi_class: SKMultiClass::Auto,
        class_weight: SKClassWeight::None,
        include_intercept,
        warm_start: false,
    }
}

fn solve_dense(
    features: &Array2<f64>,
    labels: &[usize],
    classes: usize,
    strategy: SKResolvedMultiClass,
    config: &SKLogisticConfig<f64>,
) -> SKLogisticSolution<f64> {
    let design = SKDenseDesign::new(features.view());
    let weights = vec![1.0_f64; labels.len()];
    sk_solve_logistic(&design, labels, &weights, classes, strategy, config, None).unwrap()
}

#[test]
fn lbfgs_binary_separates_the_classes() {
    let (features, labels) = separable_binary_design_with_zero_column();
    let solution = solve_dense(
        &features,
        &labels,
        2,
        SKResolvedMultiClass::Binary,
        &config(SKLogisticSolver::Lbfgs, SKLogisticPenalty::L2, 1.0, true),
    );
    assert_eq!(solution.coefficients.nrows(), 1);
    let weights = solution.coefficients.row(0);
    let intercept = solution.intercepts[0];
    for (row, &label) in labels.iter().enumerate() {
        let decision = features.row(row).dot(&weights) + intercept;
        let predicted = if decision > 0.0 { 1 } else { 0 };
        assert_eq!(predicted, label, "row {row} misclassified");
    }
}

#[test]
fn lbfgs_multinomial_fits_three_classes() {
    let (features, labels) = three_class_design();
    let solution = solve_dense(
        &features,
        &labels,
        3,
        SKResolvedMultiClass::Multinomial,
        &config(SKLogisticSolver::Lbfgs, SKLogisticPenalty::L2, 1.0, true),
    );
    assert_eq!(solution.coefficients.nrows(), 3);
    for (row, &label) in labels.iter().enumerate() {
        let decisions: Vec<f64> = (0..3)
            .map(|head| {
                features.row(row).dot(&solution.coefficients.row(head)) + solution.intercepts[head]
            })
            .collect();
        let predicted = decisions
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;
        assert_eq!(predicted, label, "row {row} misclassified");
    }
}

#[test]
fn saga_l1_drives_the_zero_column_to_exact_zero() {
    let (features, labels) = separable_binary_design_with_zero_column();
    let solution = solve_dense(
        &features,
        &labels,
        2,
        SKResolvedMultiClass::Binary,
        &config(SKLogisticSolver::Saga, SKLogisticPenalty::L1, 0.05, true),
    );
    assert_eq!(
        solution.coefficients[[0, 2]],
        0.0,
        "the L1 proximal step must zero the irrelevant column exactly"
    );
}

#[test]
fn saga_and_lbfgs_agree_under_l2() {
    let (features, labels) = separable_binary_design_with_zero_column();
    let lbfgs = solve_dense(
        &features,
        &labels,
        2,
        SKResolvedMultiClass::Binary,
        &config(SKLogisticSolver::Lbfgs, SKLogisticPenalty::L2, 1.0, true),
    );
    let saga = solve_dense(
        &features,
        &labels,
        2,
        SKResolvedMultiClass::Binary,
        &config(SKLogisticSolver::Saga, SKLogisticPenalty::L2, 1.0, true),
    );
    for column in 0..3 {
        assert_close_scalar(
            lbfgs.coefficients[[0, column]],
            saga.coefficients[[0, column]],
            1e-2,
        );
    }
}

#[test]
fn liblinear_l2_matches_lbfgs_without_intercept() {
    let (features, labels) = separable_binary_design_with_zero_column();
    let lbfgs = solve_dense(
        &features,
        &labels,
        2,
        SKResolvedMultiClass::Binary,
        &config(SKLogisticSolver::Lbfgs, SKLogisticPenalty::L2, 1.0, false),
    );
    let liblinear = solve_dense(
        &features,
        &labels,
        2,
        SKResolvedMultiClass::Binary,
        &config(
            SKLogisticSolver::Liblinear,
            SKLogisticPenalty::L2,
            1.0,
            false,
        ),
    );
    for column in 0..3 {
        assert_close_scalar(
            lbfgs.coefficients[[0, column]],
            liblinear.coefficients[[0, column]],
            1e-2,
        );
    }
}

#[test]
fn liblinear_l1_produces_a_sparse_solution() {
    let (features, labels) = separable_binary_design_with_zero_column();
    let solution = solve_dense(
        &features,
        &labels,
        2,
        SKResolvedMultiClass::Binary,
        &config(
            SKLogisticSolver::Liblinear,
            SKLogisticPenalty::L1,
            0.05,
            true,
        ),
    );
    assert_eq!(solution.coefficients[[0, 2]], 0.0);
}

#[test]
fn one_versus_rest_fits_one_head_per_class() {
    let (features, _) = three_class_design();
    let labels = vec![0, 1, 2, 0, 1, 2];
    let solution = solve_dense(
        &features,
        &labels,
        3,
        SKResolvedMultiClass::OneVersusRest,
        &config(SKLogisticSolver::Lbfgs, SKLogisticPenalty::L2, 1.0, true),
    );
    assert_eq!(solution.coefficients.nrows(), 3);
    assert_eq!(solution.intercepts.len(), 3);
}

#[test]
fn sparse_and_dense_agree() {
    let (features, labels) = separable_binary_design_with_zero_column();
    let sparse = to_csr(&features);
    let weights = vec![1.0_f64; labels.len()];
    let config = config(SKLogisticSolver::Saga, SKLogisticPenalty::L1, 0.1, true);
    let dense_solution = solve_dense(&features, &labels, 2, SKResolvedMultiClass::Binary, &config);
    let sparse_solution = sk_solve_logistic(
        &SKSparseDesign::new(sparse.view()),
        &labels,
        &weights,
        2,
        SKResolvedMultiClass::Binary,
        &config,
        None,
    )
    .unwrap();
    for column in 0..3 {
        assert_close_scalar(
            dense_solution.coefficients[[0, column]],
            sparse_solution.coefficients[[0, column]],
            1e-6,
        );
    }
}

#[test]
fn a_tiny_budget_reports_non_convergence() {
    let (features, labels) = separable_binary_design_with_zero_column();
    let mut config = config(SKLogisticSolver::Saga, SKLogisticPenalty::L1, 0.05, true);
    config.maximum_number_of_iterations = 1;
    config.tolerance = 1e-14;
    let design = SKDenseDesign::new(features.view());
    let weights = vec![1.0_f64; labels.len()];
    let result = sk_solve_logistic(
        &design,
        &labels,
        &weights,
        2,
        SKResolvedMultiClass::Binary,
        &config,
        None,
    );
    assert!(result.is_err());
}

#[test]
fn solvers_work_on_f32() {
    let features = array![[-2.0_f32, -1.0], [1.0, 2.0]];
    let labels = vec![0usize, 1];
    let config = SKLogisticConfig {
        alpha: 1.0_f32,
        tolerance: 1e-6,
        maximum_number_of_iterations: 5_000,
        penalty: SKLogisticPenalty::L2,
        l1_ratio: 1.0,
        solver: SKLogisticSolver::Lbfgs,
        multi_class: SKMultiClass::Auto,
        class_weight: SKClassWeight::None,
        include_intercept: true,
        warm_start: false,
    };
    let design = SKDenseDesign::new(features.view());
    let weights = vec![1.0_f32; 2];
    let solution = sk_solve_logistic(
        &design,
        &labels,
        &weights,
        2,
        SKResolvedMultiClass::Binary,
        &config,
        None,
    )
    .unwrap();
    assert_eq!(solution.coefficients.nrows(), 1);
    let _ = Array1::<f32>::zeros(1);
}
