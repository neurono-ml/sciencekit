//! Companion tests for the dense randomized range finder (spec `randomized-svd`).

use ndarray::Array2;
use sciencekit_common::SKDataView;

use super::randomized::sk_randomized_svd;

/// Deterministic pseudo-random fixture values in `[-1, 1]`.
pub(crate) fn fixture_values(count: usize, seed: u64) -> Vec<f64> {
    let mut state = seed;
    (0..count)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((state >> 11) as f64 / (1_u64 << 53) as f64) * 2.0 - 1.0
        })
        .collect()
}

/// A dense fixture of the given shape.
pub(crate) fn fixture_matrix(rows: usize, columns: usize, seed: u64) -> Array2<f64> {
    Array2::from_shape_vec((rows, columns), fixture_values(rows * columns, seed)).unwrap()
}

/// A rank-limited fixture: the product of a `rows × rank` and a `rank × columns`
/// matrix.
pub(crate) fn fixture_low_rank_matrix(
    rows: usize,
    columns: usize,
    rank: usize,
    seed: u64,
) -> Array2<f64> {
    let left = fixture_matrix(rows, rank, seed);
    let right = fixture_matrix(rank, columns, seed + 1);
    left.dot(&right)
}

/// The relative Frobenius error between a matrix and its reconstruction.
pub(crate) fn relative_error(reference: &Array2<f64>, approximation: &Array2<f64>) -> f64 {
    let difference = reference - approximation;
    let numerator = difference
        .iter()
        .map(|value| value * value)
        .sum::<f64>()
        .sqrt();
    let denominator = reference
        .iter()
        .map(|value| value * value)
        .sum::<f64>()
        .sqrt();
    numerator / denominator
}

#[test]
fn randomized_dense_reconstructs_a_low_rank_matrix() {
    let matrix = fixture_low_rank_matrix(60, 20, 3, 5);
    let result = sk_randomized_svd(SKDataView::Dense(matrix.view()), 3, 2, 5, 7, None)
        .expect("the decomposition must succeed");

    let reconstruction = result.reconstruct();
    let error = relative_error(&matrix, &reconstruction);
    assert!(error < 1e-8, "reconstruction error {error} is too large");
}

#[test]
fn randomized_dense_power_iterations_improve_a_decaying_spectrum() {
    let matrix = fixture_low_rank_matrix(80, 30, 12, 21);
    let single = sk_randomized_svd(SKDataView::Dense(matrix.view()), 4, 1, 6, 3, None)
        .expect("the decomposition must succeed");
    let repeated = sk_randomized_svd(SKDataView::Dense(matrix.view()), 4, 7, 6, 3, None)
        .expect("the decomposition must succeed");

    let single_error = relative_error(&matrix, &single.reconstruct());
    let repeated_error = relative_error(&matrix, &repeated.reconstruct());
    assert!(
        repeated_error <= single_error + 1e-12,
        "power iterations must not increase the error: {single_error} -> {repeated_error}"
    );
}

#[test]
fn randomized_dense_projects_a_wide_matrix_without_oversampling_overflow() {
    let matrix = fixture_matrix(10, 40, 9);
    let result = sk_randomized_svd(SKDataView::Dense(matrix.view()), 8, 2, 4, 1, None)
        .expect("the decomposition must succeed");
    assert_eq!(result.u.dim(), (10, 8));
    assert_eq!(result.v.dim(), (40, 8));
}

#[test]
fn randomized_dense_result_shapes_match_the_input() {
    let matrix = fixture_low_rank_matrix(30, 12, 4, 13);
    let result = sk_randomized_svd(SKDataView::Dense(matrix.view()), 5, 2, 3, 8, None)
        .expect("the decomposition must succeed");
    assert_eq!(result.u.dim(), (30, 5));
    assert_eq!(result.v.dim(), (12, 5));
    assert_eq!(result.singular_values.len(), 5);
}

#[test]
fn randomized_dense_singular_values_are_non_negative_and_sorted() {
    let matrix = fixture_low_rank_matrix(30, 12, 4, 17);
    let result = sk_randomized_svd(SKDataView::Dense(matrix.view()), 5, 2, 3, 8, None)
        .expect("the decomposition must succeed");
    for pair in result.singular_values.windows(2) {
        assert!(
            pair[0] >= pair[1],
            "the singular values must be non-increasing"
        );
    }
    assert!(result.singular_values.iter().all(|value| *value >= 0.0));
}

#[test]
fn randomized_dense_sign_convention_makes_the_pivot_entry_positive() {
    let matrix = fixture_low_rank_matrix(30, 12, 4, 23);
    let result = sk_randomized_svd(SKDataView::Dense(matrix.view()), 5, 2, 3, 8, None)
        .expect("the decomposition must succeed");
    for component in 0..result.u.ncols() {
        let column = result.u.column(component);
        let pivot = column
            .iter()
            .enumerate()
            .max_by(|first, second| {
                first
                    .1
                    .abs()
                    .partial_cmp(&second.1.abs())
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| second.0.cmp(&first.0))
            })
            .map(|(index, value)| (index, *value))
            .expect("the column is not empty");
        assert!(
            pivot.1 >= 0.0,
            "the largest-magnitude entry of a singular vector must be positive"
        );
    }
}

#[test]
fn randomized_dense_repeated_calls_return_identical_results() {
    let matrix = fixture_low_rank_matrix(30, 12, 4, 29);
    let first = sk_randomized_svd(SKDataView::Dense(matrix.view()), 5, 2, 3, 8, None)
        .expect("the decomposition must succeed");
    let second = sk_randomized_svd(SKDataView::Dense(matrix.view()), 5, 2, 3, 8, None)
        .expect("the decomposition must succeed");
    assert_eq!(first.u, second.u);
    assert_eq!(first.singular_values, second.singular_values);
    assert_eq!(first.v, second.v);
}
