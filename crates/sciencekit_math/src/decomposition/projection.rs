//! The seeded Gaussian test basis of the randomized range finder.

use ndarray::Array2;
use sciencekit_common::{SK_RANDOM_DOMAIN_PROJECTION, SKFloat, SKRandomGenerator, sk_derive_seed};

/// Builds the seeded Gaussian test basis `Ω` of the randomized range finder.
///
/// The matrix has `rows` rows and `columns` columns and is filled with
/// standard-normal draws from the shared [`SKRandomGenerator`], seeded through
/// the projection domain of [`sk_derive_seed`]. The stream is a pure function
/// of the seed: the same seed and shape always yield the same basis, on any
/// thread count and in any order of calls.
pub fn sk_seeded_projection<F: SKFloat>(
    rows: usize,
    columns: usize,
    random_seed: u64,
) -> Array2<F> {
    let mut generator =
        SKRandomGenerator::new(sk_derive_seed(random_seed, SK_RANDOM_DOMAIN_PROJECTION, 0));
    Array2::from_shape_fn((rows, columns), |_| {
        <F as num_traits::NumCast>::from(generator.normal()).unwrap_or_else(F::zero)
    })
}
