//! Companion tests for the neighbour weighting schemes
//! (spec `nearest-neighbor-estimators`).

use super::SKNeighborsWeights;
use super::weights::sk_weighted_average;

fn assert_close(left: f64, right: f64) {
    assert!(
        (left - right).abs() < 1e-12,
        "expected {right}, found {left}"
    );
}

#[test]
fn weights_uniform_gives_every_neighbor_the_same_weight() {
    let weights = SKNeighborsWeights::Uniform.weights(&[1.0_f64, 0.5, 3.0, 7.0]);
    assert_eq!(weights, vec![1.0, 1.0, 1.0, 1.0]);
}

#[test]
fn weights_distance_uses_the_inverse_distance() {
    let weights = SKNeighborsWeights::Distance.weights(&[1.0_f64, 2.0, 4.0]);
    assert_close(weights[0], 1.0);
    assert_close(weights[1], 0.5);
    assert_close(weights[2], 0.25);
}

#[test]
fn weights_distance_keeps_only_zero_distance_neighbors_for_duplicates() {
    let weights = SKNeighborsWeights::Distance.weights(&[0.0_f64, 0.0, 2.0]);
    assert_eq!(weights, vec![1.0, 1.0, 0.0]);
}

#[test]
fn weights_distance_zero_distance_may_be_in_any_position() {
    let weights = SKNeighborsWeights::Distance.weights(&[3.0_f64, 0.0, 1.0]);
    assert_eq!(weights, vec![0.0, 1.0, 0.0]);
}

#[test]
fn weights_weighted_average_matches_the_hand_computed_value() {
    let values = [2.0_f64, 4.0, 6.0];
    let weights = SKNeighborsWeights::Distance.weights(&[1.0_f64, 2.0, 4.0]);
    // (1·2 + 0.5·4 + 0.25·6) / (1 + 0.5 + 0.25) = 5.5 / 1.75
    assert_close(sk_weighted_average(&weights, &values), 5.5 / 1.75);
}

#[test]
fn weights_zero_distance_makes_the_average_the_duplicate_mean() {
    let values = [10.0_f64, 20.0, 30.0];
    let weights = SKNeighborsWeights::Distance.weights(&[0.0_f64, 0.0, 5.0]);
    assert_close(sk_weighted_average(&weights, &values), 15.0);
}

#[test]
fn weights_uniform_average_is_the_arithmetic_mean() {
    let weights = SKNeighborsWeights::Uniform.weights(&[1.0_f64, 2.0, 9.0]);
    assert_close(sk_weighted_average(&weights, &[2.0_f64, 4.0, 6.0]), 4.0);
}
