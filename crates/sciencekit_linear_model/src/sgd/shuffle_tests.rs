//! Companion tests for the seeded shuffler (spec `stochastic-gradient-solver`,
//! decision 6).

use ndarray::{Array1, Array2};
use sciencekit_common::SKFloat;

use super::config::SKSGDConfig;
use super::epoch::{sk_run_dense_epoch, sk_run_sparse_epoch};
use super::learning_rate::schedule::SKLearningRateSchedule;
use super::loss::SKSGDRegressorLoss;
use super::loss::regressor::sk_regressor_loss_adapter;
use super::shuffle::SKRandomGenerator;
use super::state::SKLearningState;
use crate::test_datasets::to_csr;

#[test]
fn the_same_seed_produces_the_same_order() {
    let first = SKRandomGenerator::new(7).shuffled_indices(64);
    let second = SKRandomGenerator::new(7).shuffled_indices(64);
    assert_eq!(first, second);
}

#[test]
fn different_seeds_produce_different_orders() {
    let first = SKRandomGenerator::new(1).shuffled_indices(64);
    let second = SKRandomGenerator::new(2).shuffled_indices(64);
    assert_ne!(first, second);
}

#[test]
fn empty_and_single_row_orders_are_trivial() {
    assert!(SKRandomGenerator::new(0).shuffled_indices(0).is_empty());
    assert_eq!(SKRandomGenerator::new(0).shuffled_indices(1), vec![0]);
}

#[test]
fn the_order_covers_every_row_exactly_once() {
    let mut order = SKRandomGenerator::new(42).shuffled_indices(100);
    order.sort_unstable();
    assert_eq!(order, (0..100).collect::<Vec<usize>>());
}

fn epoch_state<F: SKFloat>(config: &SKSGDConfig<F>, features: usize) -> SKLearningState<F> {
    let schedule = SKLearningRateSchedule::new(
        config.learning_rate,
        config.initial_learning_rate,
        config.inverse_scaling_power,
        config.alpha,
        F::one(),
    );
    SKLearningState::new(Array1::zeros(features), F::zero(), schedule, false)
}

#[test]
fn the_seeded_order_applies_identically_to_dense_and_sparse_designs() {
    // One order, generated from the seed and applied to a dense design and its
    // CSR mirror: the representation only changes how a row is yielded, never
    // which row is visited.
    let rows = 37;
    let order = SKRandomGenerator::new(11).shuffled_indices(rows);
    let mut sorted = order.clone();
    sorted.sort_unstable();
    assert_eq!(sorted, (0..rows).collect::<Vec<usize>>());

    let mut values = Vec::with_capacity(rows * 2);
    for row in 0..rows {
        values.push(((row * 7) % 11) as f64 / 11.0);
        values.push(((row * 5) % 13) as f64 / 13.0);
    }
    let design = Array2::from_shape_vec((rows, 2), values).unwrap();
    let targets = Array1::from_iter((0..rows).map(|row| ((row * 3) % 7) as f64));
    let weights = vec![1.0; rows];
    let config = SKSGDConfig::<f64>::default();
    let loss = sk_regressor_loss_adapter(SKSGDRegressorLoss::SquaredError, 0.0);

    let mut dense_state = epoch_state(&config, 2);
    sk_run_dense_epoch(
        &mut dense_state,
        design.view(),
        targets.view(),
        &weights,
        &loss,
        &config,
        &order,
    )
    .unwrap();
    let sparse = to_csr(&design);
    let mut sparse_state = epoch_state(&config, 2);
    sk_run_sparse_epoch(
        &mut sparse_state,
        sparse.view(),
        targets.view(),
        &weights,
        &loss,
        &config,
        &order,
    )
    .unwrap();
    for column in 0..2 {
        assert!((dense_state.weights[column] - sparse_state.weights[column]).abs() < 1e-12);
    }
    assert!((dense_state.intercept - sparse_state.intercept).abs() < 1e-12);
    assert_eq!(dense_state.step, rows);
}

#[test]
fn next_index_stays_inside_the_range() {
    let mut generator = SKRandomGenerator::new(5);
    for bound in [1, 2, 3, 7, 100] {
        for _ in 0..50 {
            assert!(generator.next_index(bound) < bound.max(1));
        }
    }
}

#[test]
fn the_generator_stream_is_deterministic() {
    let mut first = SKRandomGenerator::new(123);
    let mut second = SKRandomGenerator::new(123);
    for _ in 0..10 {
        assert_eq!(first.next_u64(), second.next_u64());
    }
}
