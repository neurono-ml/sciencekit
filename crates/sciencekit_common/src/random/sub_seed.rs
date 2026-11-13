//! Deterministic sub-seed derivation (spec `deterministic-randomness`,
//! decision 3).
//!
//! A single global seed is not enough when independent components run in the
//! same fit: trees, initializations, stages, folds and projections must not
//! share a stream. [`sk_derive_seed`] mixes the global seed with a domain tag
//! and a component index, and is a pure function — the same three inputs always
//! produce the same sub-seed, regardless of processing order or thread.

/// Domain tag for tree-structured components (random forests, boosting).
pub const SK_RANDOM_DOMAIN_TREE: u64 = 1;

/// Domain tag for initialization draws (centroids, weights, projections).
pub const SK_RANDOM_DOMAIN_INITIALIZATION: u64 = 2;

/// Domain tag for sequential stages of one algorithm.
pub const SK_RANDOM_DOMAIN_STAGE: u64 = 3;

/// Domain tag for cross-validation folds.
pub const SK_RANDOM_DOMAIN_FOLD: u64 = 4;

/// Domain tag for randomized projections (randomized SVD, hashing).
pub const SK_RANDOM_DOMAIN_PROJECTION: u64 = 5;

/// The SplitMix64 finalizer: an invertible bit mixer with full avalanche.
fn mix_bits(value: u64) -> u64 {
    let mut mixed = value;
    mixed = (mixed ^ (mixed >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    mixed = (mixed ^ (mixed >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    mixed ^ (mixed >> 31)
}

/// Derives a component sub-seed from a global seed, a domain tag and an index.
///
/// The domain is folded in first and the index second, each through a distinct
/// golden-ratio multiplier followed by the SplitMix64 finalizer, so a
/// component's seed depends on all three inputs and on nothing else. Because
/// the function keeps no state, deriving the seeds in a different order or on a
/// different number of threads yields the same value per component.
///
/// ```
/// use sciencekit_common::{SK_RANDOM_DOMAIN_TREE, sk_derive_seed};
///
/// let first = sk_derive_seed(42, SK_RANDOM_DOMAIN_TREE, 0);
/// let second = sk_derive_seed(42, SK_RANDOM_DOMAIN_TREE, 1);
/// assert_ne!(first, second);
/// ```
pub fn sk_derive_seed(global_seed: u64, domain: u64, index: u64) -> u64 {
    let with_domain = global_seed ^ domain.wrapping_mul(0xD1B5_4A32_D192_ED03);
    let mixed = mix_bits(with_domain);
    mix_bits(mixed ^ index.wrapping_mul(0x9E37_79B9_7F4A_7C15))
}
