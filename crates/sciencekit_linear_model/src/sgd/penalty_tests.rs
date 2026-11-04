//! Companion tests for the penalty selection and the proximal update (spec
//! `stochastic-gradient-solver`, decision 3).

use ndarray::array;

use super::penalty::{SKSGDPenalty, sk_apply_proximal_penalty, sk_soft_threshold};

#[test]
fn l2_penalizes_only_the_squared_norm() {
    let penalty = SKSGDPenalty::L2;
    assert_eq!(penalty.l1_coefficient(0.4_f64, 0.9), 0.0);
    assert_eq!(penalty.l2_coefficient(0.4_f64, 0.9), 0.4);
}

#[test]
fn l1_penalizes_only_the_absolute_value_even_with_an_l1_ratio() {
    let penalty = SKSGDPenalty::L1;
    assert_eq!(penalty.l1_coefficient(0.4_f64, 0.9), 0.4);
    assert_eq!(penalty.l2_coefficient(0.4_f64, 0.9), 0.0);
}

#[test]
fn elastic_net_splits_alpha_by_the_l1_ratio() {
    let penalty = SKSGDPenalty::ElasticNet;
    assert!((penalty.l1_coefficient(1.0_f64, 0.25) - 0.25).abs() < 1e-12);
    assert!((penalty.l2_coefficient(1.0_f64, 0.25) - 0.75).abs() < 1e-12);
}

#[test]
fn none_penalizes_nothing() {
    let penalty = SKSGDPenalty::None;
    assert_eq!(penalty.l1_coefficient(1.0_f64, 0.5), 0.0);
    assert_eq!(penalty.l2_coefficient(1.0_f64, 0.5), 0.0);
}

#[test]
fn l2_is_the_default_penalty() {
    assert_eq!(SKSGDPenalty::default(), SKSGDPenalty::L2);
}

#[test]
fn soft_threshold_zeroes_inside_the_band_and_shifts_outside_it() {
    assert_eq!(sk_soft_threshold(0.5_f64, 1.0), 0.0);
    assert_eq!(sk_soft_threshold(-0.5_f64, 1.0), 0.0);
    assert!((sk_soft_threshold(2.5_f64, 1.0) - 1.5).abs() < 1e-12);
    assert!((sk_soft_threshold(-2.5_f64, 1.0) + 1.5).abs() < 1e-12);
    assert_eq!(sk_soft_threshold(1.0_f64, 1.0), 0.0);
}

#[test]
fn the_proximal_update_shrinks_then_thresholds() {
    let mut weights = array![2.0_f64, -2.0, 0.1];
    // L2 decay 0.1 then L1 threshold 0.5: 2 → 1.8 → 1.3; -2 → -1.8 → -1.3;
    // 0.1 → 0.09 → 0.
    sk_apply_proximal_penalty(weights.view_mut(), 1.0, 0.5, 0.1);
    assert!((weights[0] - 1.3).abs() < 1e-12);
    assert!((weights[1] + 1.3).abs() < 1e-12);
    assert_eq!(weights[2], 0.0);
}

#[test]
fn an_l1_penalty_produces_exact_zeros() {
    let mut weights = array![0.3_f64, -0.2, 0.05];
    sk_apply_proximal_penalty(weights.view_mut(), 1.0, 0.5, 0.0);
    assert_eq!(weights, array![0.0, 0.0, 0.0]);
}

#[test]
fn an_absent_penalty_leaves_the_weights_untouched() {
    let mut weights = array![0.3_f64, -0.2];
    sk_apply_proximal_penalty(weights.view_mut(), 1.0, 0.0, 0.0);
    assert_eq!(weights, array![0.3, -0.2]);
}
