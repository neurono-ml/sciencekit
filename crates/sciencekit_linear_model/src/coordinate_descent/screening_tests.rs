//! Gap-safe screening tests: correctness by construction, the strong set and
//! the KKT reactivation guarantee (spec `coordinate-descent-solver`).

use super::access::SKDenseColumns;
use super::config::{SKCoordinateDescentConfig, SKSelection};
use super::screening::{
    sk_dual_feasible_scale, sk_gap_safe_radius, sk_is_safely_zero, sk_kkt_violations,
};
use super::solver::{sk_solve_all_alphas, sk_solve_alpha_with_initial_active};
use super::test_datasets::{TIGHT_TOLERANCE, assert_close, orthogonal_design};

/// A configuration with screening toggled and a tight tolerance.
fn config(screening: bool) -> SKCoordinateDescentConfig<f64> {
    SKCoordinateDescentConfig {
        tolerance: TIGHT_TOLERANCE,
        maximum_number_of_iterations: 100_000,
        positive: false,
        selection: SKSelection::Cyclic,
        screening,
    }
}

/// Screening on and off yield the same solution.
#[test]
fn screening_matches_unscreened() {
    let (features, targets) = orthogonal_design();
    let access = SKDenseColumns::new(features.view());
    let screened =
        sk_solve_all_alphas(&access, targets.view(), &[0.5], 1.0, true, &config(true)).unwrap();
    let unscreened =
        sk_solve_all_alphas(&access, targets.view(), &[0.5], 1.0, true, &config(false)).unwrap();
    assert_close(
        &screened.coefficients.row(0).to_vec(),
        &unscreened.coefficients.row(0).to_vec(),
        1e-10,
    );
    assert!(screened.screened_count >= 1);
    assert_eq!(unscreened.screened_count, 0);
}

/// A coordinate deliberately left out of the initial active set is recovered by
/// the KKT reactivation pass.
#[test]
fn kkt_reactivation_recovers_a_screened_coordinate() {
    let (features, targets) = orthogonal_design();
    let access = SKDenseColumns::new(features.view());
    // Only the first coordinate is active; the second belongs in the active set
    // (its gradient exceeds the threshold) and must be reactivated.
    let solution = sk_solve_alpha_with_initial_active(
        &access,
        targets.view(),
        0.5,
        1.0,
        true,
        &config(true),
        vec![true, false, false],
    )
    .unwrap();
    assert_close(&solution.coefficients.to_vec(), &[1.5, -0.5, 0.0], 1e-8);
}

/// The gap-safe radius is `√(2·gap/n)` and never negative.
#[test]
fn gap_safe_radius_is_well_formed() {
    // gap = 1, n = 8 → √(2/8) = 0.5.
    assert!((sk_gap_safe_radius(1.0_f64, 8) - 0.5).abs() < 1e-12);
    assert_eq!(sk_gap_safe_radius(-3.0_f64, 8), 0.0);
}

/// The dual-feasible scale returns one when the gradient already respects the
/// threshold and the ratio otherwise.
#[test]
fn dual_feasible_scale_is_clamped() {
    assert_eq!(sk_dual_feasible_scale(0.5_f64, 1.0), 1.0);
    assert!((sk_dual_feasible_scale(3.0_f64, 1.0) - 3.0).abs() < 1e-12);
    // A zero threshold is the unconstrained case: no scaling.
    assert_eq!(sk_dual_feasible_scale(3.0_f64, 0.0), 1.0);
}

/// The safe-zero test screens only when the whole ball stays below the
/// threshold.
#[test]
fn safe_zero_test_is_conservative() {
    assert!(sk_is_safely_zero(0.1_f64, 1.0, 0.1, 1.0));
    assert!(!sk_is_safely_zero(0.9_f64, 1.0, 0.1, 1.0));
}

/// The KKT check reports only inactive coordinates above the threshold.
#[test]
fn kkt_violations_reports_inactive_violators() {
    let gradient = [0.2_f64, 1.5, 5.0];
    let active = [true, false, false];
    let violators = sk_kkt_violations(&gradient, &active, 1.0, 1e-9);
    assert_eq!(violators, vec![1, 2]);
}
