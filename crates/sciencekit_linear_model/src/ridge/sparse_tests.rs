//! Sparse (`SparseCg`) ridge tests: cross-solver agreement, representation
//! dispatch and the non-convergence path (spec `ridge-regression`).

use sciencekit_common::builders::SKBuilder;
use sciencekit_common::{SKBackendKind, SKBuilderState, SKError, SKSupervisedFit};

use super::builder::SKRidgeSolver;
use super::core_implementation::SKRidge;
use super::sparse_test_datasets::{SPARSE_AGREEMENT_TOLERANCE, sparse_cg, to_csr};
use super::test_datasets::{assert_close, assert_close_scalar, one_feature, two_feature};

/// A sparse fit matches the dense `Cholesky` reference within tolerance.
#[test]
fn sparse_cg_matches_dense_cholesky() {
    let (dense, targets) = two_feature();
    let sparse = to_csr(&dense);
    let reference = SKRidge::build()
        .alpha(1.0)
        .solver(SKRidgeSolver::Cholesky)
        .build()
        .unwrap()
        .fit(dense.view(), targets.view())
        .unwrap();
    let model = sparse_cg(1.0).fit(sparse.view(), targets.view()).unwrap();
    assert_eq!(model.solver(), SKRidgeSolver::SparseCg);
    assert_close(
        &model.coefficient_for(0).to_vec(),
        &reference.coefficient_for(0).to_vec(),
        SPARSE_AGREEMENT_TOLERANCE,
    );
    assert_close_scalar(
        model.intercept_for(0),
        reference.intercept_for(0),
        SPARSE_AGREEMENT_TOLERANCE,
    );
}

/// `Automatic` resolves a sparse design to `SparseCg`.
#[test]
fn sparse_automatic_resolves_to_sparse_cg() {
    let (dense, targets) = two_feature();
    let sparse = to_csr(&dense);
    let model = SKRidge::build()
        .alpha(1.0)
        .tolerance(1e-12)
        .build()
        .unwrap()
        .fit(sparse.view(), targets.view())
        .unwrap();
    assert_eq!(model.solver(), SKRidgeSolver::SparseCg);
}

/// An explicit `Cholesky` accepts sparse input through the sparse Gram.
#[test]
fn sparse_cholesky_accepts_sparse_input() {
    let (dense, targets) = two_feature();
    let sparse = to_csr(&dense);
    let model = SKRidge::build()
        .alpha(1.0)
        .solver(SKRidgeSolver::Cholesky)
        .build()
        .unwrap()
        .fit(sparse.view(), targets.view())
        .unwrap();
    let reference = SKRidge::build()
        .alpha(1.0)
        .solver(SKRidgeSolver::Cholesky)
        .build()
        .unwrap()
        .fit(dense.view(), targets.view())
        .unwrap();
    assert_eq!(model.solver(), SKRidgeSolver::Cholesky);
    assert_close(
        &model.coefficient_for(0).to_vec(),
        &reference.coefficient_for(0).to_vec(),
        SPARSE_AGREEMENT_TOLERANCE,
    );
}

/// The `Svd` solver rejects sparse input with a structured error.
#[test]
fn sparse_svd_is_rejected() {
    let (dense, targets) = two_feature();
    let sparse = to_csr(&dense);
    let error = SKRidge::build()
        .alpha(1.0)
        .solver(SKRidgeSolver::Svd)
        .build()
        .unwrap()
        .fit(sparse.view(), targets.view())
        .unwrap_err();
    assert!(matches!(
        error,
        SKError::UnsupportedRepresentation {
            representation: "csr",
            ..
        }
    ));
}

/// A zero iteration budget exhausts immediately with a structured error.
#[test]
fn sparse_zero_iteration_budget_fails_to_converge() {
    let (dense, targets) = two_feature();
    let sparse = to_csr(&dense);
    // The builder validates the budget as positive, so the runtime
    // non-convergence path is exercised with a directly constructed estimator.
    let estimator = SKRidge {
        plan: SKBuilderState::new(),
        alphas: vec![1.0],
        include_intercept: true,
        solver: SKRidgeSolver::SparseCg,
        tolerance: 1e-12,
        maximum_number_of_iterations: 0,
        backend: SKBackendKind::Faer,
    };
    let error = estimator.fit(sparse.view(), targets.view()).unwrap_err();
    assert!(matches!(error, SKError::NotConverged { iterations: 0 }));
}

/// A scalar `alpha` yields one solution; anything else would be a shape defect.
#[test]
fn sparse_scalar_alpha_yields_one_solution() {
    let (dense, targets) = one_feature();
    let sparse = to_csr(&dense);
    let model = sparse_cg(1.0).fit(sparse.view(), targets.view()).unwrap();
    assert_eq!(model.coefficients().nrows(), 1);
    assert_eq!(model.intercepts().len(), 1);
}
