//! Companion TDD tests for pairwise coupling (spec `support-vector-machines`).

use super::{sk_couple_pairwise_probabilities, sk_couple_pairwise_probabilities_with_budget};

#[test]
fn coupling_recovers_exact_three_class_distribution_and_differs_from_vote_counts() {
    // Ground-truth distribution p* = [0.6, 0.3, 0.1]:
    // Pair (0, 1): P(1 | {0, 1}) = 0.3 / 0.9 = 1/3
    // Pair (0, 2): P(2 | {0, 2}) = 0.1 / 0.7 = 1/7
    // Pair (1, 2): P(2 | {1, 2}) = 0.1 / 0.4 = 1/4
    let pairwise = [1.0_f64 / 3.0, 1.0 / 7.0, 1.0 / 4.0];
    let coupled = sk_couple_pairwise_probabilities(&pairwise, 3);

    assert_eq!(coupled.len(), 3);
    assert!((coupled.iter().sum::<f64>() - 1.0).abs() < 1e-12);
    assert!((coupled[0] - 0.6).abs() < 1e-4);
    assert!((coupled[1] - 0.3).abs() < 1e-4);
    assert!((coupled[2] - 0.1).abs() < 1e-4);

    // Note normalized vote counts would be [2/3, 1/3, 0.0], proving coupling != vote ratio!
    assert!(coupled[2] > 0.05);
}

#[test]
fn coupling_respects_iteration_cap_and_maintains_normalized_simplex_output() {
    let pairwise = [0.25_f64, 0.10, 0.20];
    let (capped, executed, converged) =
        sk_couple_pairwise_probabilities_with_budget(&pairwise, 3, 1e-15, 1);

    assert_eq!(executed, 1);
    assert!(!converged);
    assert!((capped.iter().sum::<f64>() - 1.0).abs() < 1e-12);
    capped
        .iter()
        .for_each(|&prob| assert!((0.0..=1.0).contains(&prob)));

    let (full, full_iters, full_converged) =
        sk_couple_pairwise_probabilities_with_budget(&pairwise, 3, 1e-6, 100);
    assert!(full_converged);
    assert!(full_iters >= 1 && full_iters <= 100);
    assert!((full.iter().sum::<f64>() - 1.0).abs() < 1e-12);
}
