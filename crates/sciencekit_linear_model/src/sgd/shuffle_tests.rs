//! Companion tests for the seeded shuffler (spec `stochastic-gradient-solver`,
//! decision 6).

use ndarray::{Array1, Array2};
use sciencekit_common::{SKFloat, SKRandomGenerator};

use super::config::SKSGDConfig;
use super::epoch::{sk_run_dense_epoch, sk_run_sparse_epoch};
use super::learning_rate::schedule::SKLearningRateSchedule;
use super::loss::SKSGDRegressorLoss;
use super::loss::regressor::sk_regressor_loss_adapter;
use super::penalty::SKSGDPenalty;
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

    let design = Array2::from_shape_fn((rows, 2), |(row, column)| {
        if column == 0 {
            ((row * 7) % 11) as f64 / 11.0
        } else {
            ((row * 5) % 13) as f64 / 13.0
        }
    });
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
    dense_state
        .weights
        .iter()
        .zip(sparse_state.weights.iter())
        .for_each(|(&dense_value, &sparse_value)| {
            assert!((dense_value - sparse_value).abs() < 1e-12);
        });
    assert!((dense_state.intercept - sparse_state.intercept).abs() < 1e-12);
    assert_eq!(dense_state.step, rows);
}

#[test]
fn disabled_shuffle_follows_the_input_row_order() {
    // One dense epoch with shuffling disabled, checked against a hand-computed
    // sequential update (spec scenario "shuffling disabled keeps the order").
    //
    // Rows: x = [1] with y = 0, then x = [2] with y = 1; squared-error slope,
    // constant rate 0.1, no penalty. Visiting row 0 first: prediction 0 gives
    // slope 0, so nothing moves; row 1 then sees prediction 0, slope −1, so
    // w = 0.1·2 = 0.2, b = 0.1. Visiting row 1 first: (w, b) = (0.2, 0.1)
    // after it; row 0 then sees prediction 0.3, slope +0.3, so
    // w = 0.2 − 0.03 = 0.17, b = 0.1 − 0.03 = 0.07. The visiting order is the
    // coefficient trail.
    let design = Array2::from_shape_fn((2, 1), |(row, _)| if row == 1 { 2.0 } else { 1.0 });
    let targets = Array1::from_shape_fn(2, |row| if row == 1 { 1.0 } else { 0.0 });
    let weights = vec![1.0, 1.0];
    let loss = sk_regressor_loss_adapter(SKSGDRegressorLoss::SquaredError, 0.0);
    let config = SKSGDConfig::<f64> {
        alpha: 0.0,
        penalty: SKSGDPenalty::None,
        initial_learning_rate: 0.1,
        ..SKSGDConfig::default()
    };

    let input_order = (0..2).collect::<Vec<usize>>();
    let mut sequential = epoch_state(&config, 1);
    sk_run_dense_epoch(
        &mut sequential,
        design.view(),
        targets.view(),
        &weights,
        &loss,
        &config,
        &input_order,
    )
    .unwrap();
    assert!((sequential.weights[0] - 0.2).abs() < 1e-12);
    assert!((sequential.intercept - 0.1).abs() < 1e-12);
    assert_eq!(sequential.step, 2);

    // The deliberately reversed order produces a different, equally
    // predictable trail: only the row order changed.
    let mut reversed = epoch_state(&config, 1);
    sk_run_dense_epoch(
        &mut reversed,
        design.view(),
        targets.view(),
        &weights,
        &loss,
        &config,
        &[1, 0],
    )
    .unwrap();
    assert!((reversed.weights[0] - 0.17).abs() < 1e-12);
    assert!((reversed.intercept - 0.07).abs() < 1e-12);
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
