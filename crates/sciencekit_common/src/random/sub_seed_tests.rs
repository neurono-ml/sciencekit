//! Companion tests for sub-seed derivation (spec `deterministic-randomness`).

use std::collections::HashSet;

use super::sub_seed::{
    SK_RANDOM_DOMAIN_FOLD, SK_RANDOM_DOMAIN_INITIALIZATION, SK_RANDOM_DOMAIN_PROJECTION,
    SK_RANDOM_DOMAIN_STAGE, SK_RANDOM_DOMAIN_TREE, sk_derive_seed,
};

#[test]
fn golden_vectors_pin_the_derivation() {
    let cases = [
        ((0, SK_RANDOM_DOMAIN_TREE, 0), 0x4E96_155E_5F0A_1C3F),
        ((0, SK_RANDOM_DOMAIN_TREE, 1), 0xD5E2_6C2D_02E0_884E),
        (
            (42, SK_RANDOM_DOMAIN_INITIALIZATION, 7),
            0x4CF8_ECE9_FDC6_8C45,
        ),
        (
            (0xDEAD_BEEF, SK_RANDOM_DOMAIN_STAGE, 99),
            0x615C_547A_38A7_1465,
        ),
        ((7, SK_RANDOM_DOMAIN_FOLD, 0), 0x1AA5_A20D_FA85_91FC),
        ((7, SK_RANDOM_DOMAIN_PROJECTION, 0), 0x5AD3_B511_7F48_35E0),
    ];
    for ((global, domain, index), expected) in cases {
        assert_eq!(
            sk_derive_seed(global, domain, index),
            expected,
            "global {global}, domain {domain}, index {index}"
        );
    }
}

#[test]
fn derivation_is_pure_and_reproducible() {
    let first = sk_derive_seed(123, SK_RANDOM_DOMAIN_TREE, 4);
    let second = sk_derive_seed(123, SK_RANDOM_DOMAIN_TREE, 4);
    assert_eq!(first, second);
}

#[test]
fn every_component_index_gets_a_distinct_seed() {
    let seeds: HashSet<u64> = (0..1_024)
        .map(|index| sk_derive_seed(2024, SK_RANDOM_DOMAIN_TREE, index))
        .collect();
    assert_eq!(seeds.len(), 1_024);
}

#[test]
fn different_domains_do_not_share_a_stream() {
    let domains = [
        SK_RANDOM_DOMAIN_TREE,
        SK_RANDOM_DOMAIN_INITIALIZATION,
        SK_RANDOM_DOMAIN_STAGE,
        SK_RANDOM_DOMAIN_FOLD,
        SK_RANDOM_DOMAIN_PROJECTION,
    ];
    let unique: HashSet<u64> = domains
        .iter()
        .map(|&domain| sk_derive_seed(11, domain, 3))
        .collect();
    assert_eq!(unique.len(), domains.len());
}

#[test]
fn different_global_seeds_change_every_component_seed() {
    let first: Vec<u64> = (0..64)
        .map(|index| sk_derive_seed(1, SK_RANDOM_DOMAIN_FOLD, index))
        .collect();
    let second: Vec<u64> = (0..64)
        .map(|index| sk_derive_seed(2, SK_RANDOM_DOMAIN_FOLD, index))
        .collect();
    assert_ne!(first, second);
}

#[test]
fn derivation_is_order_independent() {
    // Deriving in reverse order yields the same seed per component: nothing is
    // carried between calls, so the processing order cannot leak in.
    let indices: Vec<u64> = (0..128).collect();
    let in_order: Vec<u64> = indices
        .iter()
        .map(|&index| sk_derive_seed(5, SK_RANDOM_DOMAIN_STAGE, index))
        .collect();
    let reversed: Vec<u64> = indices
        .iter()
        .rev()
        .map(|&index| sk_derive_seed(5, SK_RANDOM_DOMAIN_STAGE, index))
        .rev()
        .collect();
    assert_eq!(in_order, reversed);
}

#[test]
fn derivation_is_thread_count_invariant() {
    // Components running on different threads (and a different number of
    // them) still receive the per-index seed the sequential run computes.
    let sequential: Vec<u64> = (0..64)
        .map(|index| sk_derive_seed(9, SK_RANDOM_DOMAIN_PROJECTION, index))
        .collect();
    for thread_count in [1_usize, 2, 4, 8] {
        let observed = std::thread::scope(|scope| {
            let handles: Vec<_> = (0..thread_count)
                .map(|worker| {
                    scope.spawn(move || {
                        (worker..64)
                            .step_by(thread_count)
                            .map(|index| {
                                (
                                    index,
                                    sk_derive_seed(9, SK_RANDOM_DOMAIN_PROJECTION, index as u64),
                                )
                            })
                            .collect::<Vec<(usize, u64)>>()
                    })
                })
                .collect();
            let mut collected: Vec<(usize, u64)> = handles
                .into_iter()
                .flat_map(|handle| handle.join().unwrap())
                .collect();
            collected.sort_unstable();
            collected
                .into_iter()
                .map(|(_, seed)| seed)
                .collect::<Vec<u64>>()
        });
        assert_eq!(observed, sequential, "thread count {thread_count}");
    }
}
