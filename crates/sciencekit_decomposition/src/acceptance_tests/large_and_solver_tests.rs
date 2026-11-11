//! Acceptance tests for large dense matrices, wide sparse CSR matrices and the
//! solver portfolio.

use sciencekit_common::{SKBuilder, SKUnsupervisedFit};

use super::fixtures::{planted_rank_matrix, planted_wide_csr};
use crate::{SKPCA, SKPCASolver};

#[test]
fn acceptance_pca_large_dense_design_and_all_solver_branches() {
    // Tall-skinny 4_000 x 24 with k = 3 -> Auto picks CovarianceEigh.
    let tall_large = planted_rank_matrix(4_000, 24, 3, 1e-4);
    let auto_tall = SKPCA::<f64>::build()
        .number_of_components(3)
        .build()
        .unwrap()
        .fit(&tall_large)
        .unwrap();
    assert_eq!(auto_tall.resolved_solver(), SKPCASolver::CovarianceEigh);
    let tall_ratio_sum: f64 = auto_tall.explained_variance_ratio().iter().sum();
    assert!(tall_ratio_sum > 0.999);

    // Large 900 x 120 with k = 3 -> Auto picks Randomized, and agrees with Full
    // and CovarianceEigh.
    let large = planted_rank_matrix(900, 120, 3, 0.0);
    let auto_large = SKPCA::<f64>::build()
        .number_of_components(3)
        .random_seed(21)
        .build()
        .unwrap()
        .fit(&large)
        .unwrap();
    assert_eq!(auto_large.resolved_solver(), SKPCASolver::Randomized);

    let full_large = SKPCA::<f64>::build()
        .number_of_components(3)
        .svd_solver(SKPCASolver::Full)
        .build()
        .unwrap()
        .fit(&large)
        .unwrap();
    let cov_large = SKPCA::<f64>::build()
        .number_of_components(3)
        .svd_solver(SKPCASolver::CovarianceEigh)
        .build()
        .unwrap()
        .fit(&large)
        .unwrap();

    for candidate in [&auto_large, &cov_large] {
        for (left, right) in full_large
            .explained_variance()
            .iter()
            .zip(candidate.explained_variance().iter())
        {
            assert!((left - right).abs() < 1e-5);
        }
        for (left, right) in full_large
            .components()
            .iter()
            .zip(candidate.components().iter())
        {
            assert!((left - right).abs() < 1e-5);
        }
    }
}

#[test]
fn acceptance_pca_wide_sparse_design_runs_without_densification() {
    let sparse = planted_wide_csr(1_200, 320);
    let model = SKPCA::<f64>::build()
        .number_of_components(4)
        .random_seed(33)
        .build()
        .unwrap()
        .fit(sparse.view())
        .unwrap();

    assert_eq!(model.resolved_solver(), SKPCASolver::Randomized);
    assert_eq!(model.components().dim(), (4, 320));
    let projected = model.transform(sparse.view()).unwrap();
    assert_eq!(projected.dim(), (1_200, 4));
    assert!(projected.iter().all(|value| value.is_finite()));
    assert!(model.explained_variance_ratio()[0] > 0.5);
}
