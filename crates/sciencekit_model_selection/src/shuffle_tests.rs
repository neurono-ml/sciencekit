//! Seeded shuffle reproducibility (spec `model-selection-splits`, decision 3).

use std::collections::HashSet;

use super::sk_shuffle_indices;

#[test]
fn shuffle_is_reproducible_for_the_same_seed() {
    let first = sk_shuffle_indices(42, 20);
    let second = sk_shuffle_indices(42, 20);

    assert_eq!(first, second);
}

#[test]
fn shuffle_changes_the_order_for_a_different_seed() {
    let first = sk_shuffle_indices(1, 50);
    let second = sk_shuffle_indices(2, 50);

    assert_ne!(first, second);
}

#[test]
fn shuffle_returns_a_permutation_of_every_index() {
    let shuffled = sk_shuffle_indices(7, 30);
    let unique: HashSet<usize> = shuffled.iter().copied().collect();

    assert_eq!(shuffled.len(), 30);
    assert_eq!(unique.len(), 30);
    assert!(shuffled.iter().all(|&index| index < 30));
}

#[test]
fn shuffle_handles_empty_and_single_row_inputs() {
    assert!(sk_shuffle_indices(0, 0).is_empty());
    assert_eq!(sk_shuffle_indices(0, 1), vec![0]);
}
