//! Companion tests for the seeded shuffler (spec `stochastic-gradient-solver`,
//! decision 6).

use super::shuffle::SKRandomGenerator;

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

#[test]
fn dense_and_sparse_orders_agree_under_the_same_seed() {
    // The generator only sees the row count, so the order is representation
    // independent: this is what makes the dense/sparse agreement test able to
    // fix one seed.
    let dense_order = SKRandomGenerator::new(11).shuffled_indices(37);
    let sparse_order = SKRandomGenerator::new(11).shuffled_indices(37);
    assert_eq!(dense_order, sparse_order);
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
