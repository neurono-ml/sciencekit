//! Companion tests for the shared seeded generator (spec
//! `deterministic-randomness`).

use super::generator::SKRandomGenerator;

#[test]
fn golden_vectors_pin_the_splitmix64_stream() {
    // Reference outputs of the SplitMix64 finalizer: a change to the
    // constants or the shift sequence breaks these vectors by construction.
    let cases: [(u64, [u64; 5]); 4] = [
        (
            0,
            [
                0xE220_A839_7B1D_CDAF,
                0x6E78_9E6A_A1B9_65F4,
                0x06C4_5D18_8009_454F,
                0xF88B_B8A8_724C_81EC,
                0x1B39_896A_51A8_749B,
            ],
        ),
        (
            1,
            [
                0x910A_2DEC_8902_5CC1,
                0xBEEB_8DA1_658E_EC67,
                0xF893_A2EE_FB32_555E,
                0x71C1_8690_EE42_C90B,
                0x71BB_54D8_D101_B5B9,
            ],
        ),
        (
            42,
            [
                0xBDD7_3226_2FEB_6E95,
                0x28EF_E333_B266_F103,
                0x4752_6757_130F_9F52,
                0x581C_E1FF_0E4A_E394,
                0x09BC_585A_2448_23F2,
            ],
        ),
        (
            0xDEAD_BEEF,
            [
                0x4ADF_B90F_68C9_EB9B,
                0xDE58_6A31_41A1_0922,
                0x021F_BC2F_8E1C_FC1D,
                0x7466_CE73_7BE1_6790,
                0x3BFA_8764_F685_BD1C,
            ],
        ),
    ];
    for (seed, expected) in cases {
        let mut generator = SKRandomGenerator::new(seed);
        let observed: Vec<u64> = (0..expected.len()).map(|_| generator.next_u64()).collect();
        assert_eq!(observed, expected, "seed {seed:#x}");
    }
}

#[test]
fn two_generators_with_the_same_seed_produce_identical_streams() {
    let mut first = SKRandomGenerator::new(123);
    let mut second = SKRandomGenerator::new(123);
    for _ in 0..64 {
        assert_eq!(first.next_u64(), second.next_u64());
    }
}

#[test]
fn different_seeds_produce_different_streams() {
    let mut first = SKRandomGenerator::new(1);
    let mut second = SKRandomGenerator::new(2);
    assert_ne!(first.next_u64(), second.next_u64());
}

#[test]
fn a_clone_continues_the_same_stream() {
    let mut original = SKRandomGenerator::new(99);
    original.next_u64();
    let mut clone = original.clone();
    for _ in 0..16 {
        assert_eq!(original.next_u64(), clone.next_u64());
    }
}

#[test]
fn next_index_stays_inside_the_range() {
    let mut generator = SKRandomGenerator::new(5);
    for bound in [1, 2, 3, 7, 100] {
        for _ in 0..200 {
            assert!(generator.next_index(bound) < bound.max(1));
        }
    }
}

#[test]
fn the_same_seed_produces_the_same_shuffle_order() {
    let first = SKRandomGenerator::new(7).shuffled_indices(64);
    let second = SKRandomGenerator::new(7).shuffled_indices(64);
    assert_eq!(first, second);
}

#[test]
fn different_seeds_produce_different_shuffle_orders() {
    let first = SKRandomGenerator::new(1).shuffled_indices(64);
    let second = SKRandomGenerator::new(2).shuffled_indices(64);
    assert_ne!(first, second);
}

#[test]
fn a_shuffle_is_a_permutation() {
    let mut order = SKRandomGenerator::new(42).shuffled_indices(100);
    order.sort_unstable();
    assert_eq!(order, (0..100).collect::<Vec<usize>>());
}

#[test]
fn empty_and_single_element_shuffles_are_trivial() {
    assert!(SKRandomGenerator::new(0).shuffled_indices(0).is_empty());
    assert_eq!(SKRandomGenerator::new(0).shuffled_indices(1), vec![0]);
}
