//! Seeded Fisher–Yates shuffling through the shared deterministic generator.

use sciencekit_common::SKRandomGenerator;

/// Returns a seeded Fisher–Yates permutation of `0..number_of_rows`.
///
/// The permutation is a pure function of `random_seed` and
/// `number_of_rows`: the same pair always yields the same order, regardless of
/// the input representation or the calling thread. Every index appears exactly
/// once.
pub fn sk_shuffle_indices(random_seed: u64, number_of_rows: usize) -> Vec<usize> {
    SKRandomGenerator::new(random_seed).shuffled_indices(number_of_rows)
}
