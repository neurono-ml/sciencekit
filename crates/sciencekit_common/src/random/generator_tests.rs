//! Companion tests for the shared seeded generator (spec
//! `deterministic-randomness`).

use super::generator::SKRandomGenerator;

#[test]
fn the_crate_root_reexports_the_random_surface() {
    let mut generator = crate::SKRandomGenerator::new(3);
    assert_eq!(
        generator.next_u64(),
        super::generator::SKRandomGenerator::new(3).next_u64()
    );
    assert_eq!(
        crate::sk_derive_seed(1, crate::SK_RANDOM_DOMAIN_TREE, 2),
        super::sub_seed::sk_derive_seed(1, super::sub_seed::SK_RANDOM_DOMAIN_TREE, 2)
    );
    let domains = [
        crate::SK_RANDOM_DOMAIN_INITIALIZATION,
        crate::SK_RANDOM_DOMAIN_STAGE,
        crate::SK_RANDOM_DOMAIN_FOLD,
        crate::SK_RANDOM_DOMAIN_PROJECTION,
    ];
    assert!(
        domains
            .iter()
            .all(|&domain| domain != crate::SK_RANDOM_DOMAIN_TREE)
    );
}

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

#[test]
fn next_f64_is_uniform_in_the_unit_interval() {
    let mut generator = SKRandomGenerator::new(31);
    let mut sum = 0.0;
    let draws = 100_000;
    for _ in 0..draws {
        let value = generator.next_f64();
        assert!((0.0..1.0).contains(&value), "value {value} out of range");
        sum += value;
    }
    let mean = sum / f64::from(draws);
    assert!((mean - 0.5).abs() < 0.01, "mean {mean}");
}

#[test]
fn uniform_stays_inside_the_requested_interval() {
    let mut generator = SKRandomGenerator::new(32);
    for (low, high) in [(0.0, 1.0), (-3.5, 2.25), (1e6, 1e6 + 1.0)] {
        for _ in 0..10_000 {
            let value = generator.uniform(low, high);
            assert!(
                (low..high).contains(&value),
                "value {value} outside [{low}, {high})"
            );
        }
    }
}

#[test]
fn normal_draws_have_zero_mean_and_unit_variance() {
    let mut generator = SKRandomGenerator::new(33);
    let draws = 100_000;
    let observations: Vec<f64> = (0..draws).map(|_| generator.normal()).collect();
    let mean = observations.iter().sum::<f64>() / f64::from(draws);
    let variance = observations
        .iter()
        .map(|value| (value - mean) * (value - mean))
        .sum::<f64>()
        / f64::from(draws);
    assert!(mean.abs() < 0.02, "mean {mean}");
    assert!((variance - 1.0).abs() < 0.03, "variance {variance}");
}

#[test]
fn two_normal_draws_consume_one_uniform_pair() {
    // Box–Muller produces two normals per uniform pair; the second is cached,
    // so after two draws the stream sits where it would after drawing two
    // uniforms directly.
    let mut generator = SKRandomGenerator::new(34);
    let first = generator.normal();
    let second = generator.normal();
    let mut direct = SKRandomGenerator::new(34);
    direct.next_f64();
    direct.next_f64();
    assert_eq!(generator.next_u64(), direct.next_u64());
    assert_ne!(first, second, "the cached spare must differ from its pair");
}

#[test]
fn the_same_seed_produces_the_same_normal_stream() {
    let mut first = SKRandomGenerator::new(35);
    let mut second = SKRandomGenerator::new(35);
    for _ in 0..32 {
        assert_eq!(first.normal(), second.normal());
    }
}

#[test]
fn shuffling_a_slice_matches_the_index_order() {
    let mut values: Vec<usize> = (0..50).collect();
    let mut generator = SKRandomGenerator::new(36);
    generator.shuffle(&mut values);
    let expected = SKRandomGenerator::new(36).shuffled_indices(50);
    assert_eq!(values, expected);

    let mut empty: [u8; 0] = [];
    SKRandomGenerator::new(0).shuffle(&mut empty);
    let mut single = [7_u8];
    SKRandomGenerator::new(0).shuffle(&mut single);
    assert_eq!(single, [7]);
}

#[test]
fn sampling_without_replacement_returns_unique_indices_in_range() {
    let mut generator = SKRandomGenerator::new(37);
    let mut all = Vec::new();
    for count in [0, 1, 5, 64, 1_000] {
        let indices = generator.sample_indices_without_replacement(count, 20);
        assert_eq!(indices.len(), count.min(20));
        let mut sorted = indices.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), indices.len(), "duplicate indices");
        assert!(indices.iter().all(|&index| index < count));
        all.extend(indices);
    }
    assert!(!all.is_empty());
}

#[test]
fn sampling_without_replacement_is_reproducible_and_matches_a_full_sample() {
    let first = SKRandomGenerator::new(38).sample_indices_without_replacement(40, 10);
    let second = SKRandomGenerator::new(38).sample_indices_without_replacement(40, 10);
    assert_eq!(first, second);

    let mut all = SKRandomGenerator::new(39).sample_indices_without_replacement(40, 40);
    all.sort_unstable();
    assert_eq!(all, (0..40).collect::<Vec<usize>>());

    assert!(
        SKRandomGenerator::new(0)
            .sample_indices_without_replacement(0, 0)
            .is_empty()
    );
}
