//! Companion tests for `SKPCASolver` and deterministic `Auto` resolution.

use super::SKPCASolver;
use super::solvers::sk_resolve_pca_solver;

#[test]
fn pca_solver_default_is_auto_and_explicit_solvers_are_preserved() {
    assert_eq!(SKPCASolver::default(), SKPCASolver::Auto);
    assert_eq!(
        sk_resolve_pca_solver(SKPCASolver::Full, 800, 120, 5, false, false),
        SKPCASolver::Full
    );
    assert_eq!(
        sk_resolve_pca_solver(SKPCASolver::Randomized, 10, 5, 5, false, false),
        SKPCASolver::Randomized
    );
    assert_eq!(
        sk_resolve_pca_solver(SKPCASolver::CovarianceEigh, 10, 5, 2, false, false),
        SKPCASolver::CovarianceEigh
    );
}

#[test]
fn pca_solver_auto_routes_small_high_rank_covariance_and_large_designs() {
    // Small designs resolve to Full even with a small component count.
    assert_eq!(
        sk_resolve_pca_solver(SKPCASolver::Auto, 16, 6, 2, false, false),
        SKPCASolver::Full
    );
    assert_eq!(
        sk_resolve_pca_solver(SKPCASolver::Auto, 16, 6, 6, false, false),
        SKPCASolver::Full
    );

    // High component fraction (>= 80% of min(n, d)) resolves to Full even on large data.
    assert_eq!(
        sk_resolve_pca_solver(SKPCASolver::Auto, 800, 100, 80, false, false),
        SKPCASolver::Full
    );

    // Tall-skinny designs (n > 10 * d, d <= 1000, k < 0.8 * min(n, d)) use CovarianceEigh.
    assert_eq!(
        sk_resolve_pca_solver(SKPCASolver::Auto, 300, 12, 3, false, false),
        SKPCASolver::CovarianceEigh
    );
    assert_eq!(
        sk_resolve_pca_solver(SKPCASolver::Auto, 2_000, 50, 5, false, false),
        SKPCASolver::CovarianceEigh
    );

    // Large designs with a small component count use Randomized.
    assert_eq!(
        sk_resolve_pca_solver(SKPCASolver::Auto, 800, 120, 5, false, false),
        SKPCASolver::Randomized
    );

    // Sparse designs without allow_densification resolve Auto to Randomized so no
    // dense centered matrix is ever materialized implicitly.
    assert_eq!(
        sk_resolve_pca_solver(SKPCASolver::Auto, 20, 8, 2, true, false),
        SKPCASolver::Randomized
    );
}
