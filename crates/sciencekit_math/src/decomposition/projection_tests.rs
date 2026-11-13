//! Companion tests for the seeded random projection (spec `randomized-svd`).

use super::projection::sk_seeded_projection;

#[test]
fn randomized_projection_has_the_requested_shape() {
    let projection = sk_seeded_projection::<f64>(7, 3, 11);
    assert_eq!(projection.dim(), (7, 3));
}

#[test]
fn randomized_projection_is_reproducible_for_the_same_seed() {
    let first = sk_seeded_projection::<f64>(12, 4, 42);
    let second = sk_seeded_projection::<f64>(12, 4, 42);
    assert_eq!(first, second);
}

#[test]
fn randomized_projection_differs_across_seeds() {
    let first = sk_seeded_projection::<f64>(12, 4, 1);
    let second = sk_seeded_projection::<f64>(12, 4, 2);
    assert_ne!(first, second);
}

#[test]
fn randomized_projection_draws_are_finite_and_varied() {
    let projection = sk_seeded_projection::<f64>(64, 8, 7);
    assert!(projection.iter().all(|value| value.is_finite()));
    let minimum = projection.iter().copied().fold(f64::INFINITY, f64::min);
    let maximum = projection.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    assert!(maximum - minimum > 1.0, "the basis must not be degenerate");
}

#[test]
fn randomized_projection_is_stable_for_the_single_precision_scalar() {
    let projection = sk_seeded_projection::<f32>(5, 2, 3);
    assert_eq!(projection.dim(), (5, 2));
    assert!(projection.iter().all(|value| value.is_finite()));
}
