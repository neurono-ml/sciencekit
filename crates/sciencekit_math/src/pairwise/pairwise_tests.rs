//! TDD tests for the pairwise-distance kernels (spec `pairwise-distances`).

use ndarray::{Array2, array};

use super::{
    sk_cosine_distance_matrix, sk_euclidean_distance_matrix, sk_manhattan_distance_matrix,
    sk_squared_euclidean_distance_matrix,
};

/// A known squared-Euclidean matrix matches the direct per-pair computation.
#[test]
fn squared_euclidean_matches_direct_computation() {
    let left: Array2<f64> = array![[0.0, 0.0], [3.0, 4.0]];
    let right: Array2<f64> = array![[1.0, 1.0]];
    let distance = sk_squared_euclidean_distance_matrix(&left.view(), &right.view());
    // dist([0,0],[1,1]) = 2 ; dist([3,4],[1,1]) = 4 + 9 = 13.
    assert!((distance[[0, 0]] - 2.0).abs() < 1e-12);
    assert!((distance[[1, 0]] - 13.0).abs() < 1e-12);
}

/// The distance of a set with itself is symmetric, with a zero diagonal.
#[test]
fn squared_euclidean_self_is_symmetric_with_zero_diagonal() {
    let points: Array2<f64> = array![[0.0, 0.0], [1.0, 2.0], [3.0, 5.0]];
    let distance = sk_squared_euclidean_distance_matrix(&points.view(), &points.view());
    assert_eq!(distance.diag(), array![0.0, 0.0, 0.0].view());
    assert!((distance[[0, 1]] - distance[[1, 0]]).abs() < 1e-12);
    assert!((distance[[1, 2]] - distance[[2, 1]]).abs() < 1e-12);
}

/// All entries of the squared form are non-negative.
#[test]
fn squared_euclidean_is_non_negative() {
    let points: Array2<f64> = array![[-1.0, 2.0], [3.0, -4.0], [0.5, 0.5]];
    let distance = sk_squared_euclidean_distance_matrix(&points.view(), &points.view());
    assert!(distance.iter().all(|&value| value >= 0.0));
}

/// Euclidean distance is the square root of the squared form.
#[test]
fn euclidean_is_sqrt_of_squared() {
    let left: Array2<f64> = array![[0.0, 0.0], [3.0, 4.0]];
    let right: Array2<f64> = array![[0.0, 0.0]];
    let euclidean = sk_euclidean_distance_matrix(&left.view(), &right.view());
    let squared = sk_squared_euclidean_distance_matrix(&left.view(), &right.view());
    for (e, s) in euclidean.iter().zip(squared.iter()) {
        assert!((e - s.sqrt()).abs() < 1e-12);
    }
    // dist([3,4],[0,0]) = 5.
    assert!((euclidean[[1, 0]] - 5.0).abs() < 1e-12);
}

/// Manhattan equals the sum of absolute per-coordinate differences.
#[test]
fn manhattan_is_l1_sum_of_absolute_differences() {
    let left: Array2<f64> = array![[1.0, 2.0, 3.0]];
    let right: Array2<f64> = array![[4.0, -1.0, 2.0]];
    let manhattan = sk_manhattan_distance_matrix(&left.view(), &right.view());
    // |1-4| + |2-(-1)| + |3-2| = 3 + 3 + 1 = 7.
    assert!((manhattan[[0, 0]] - 7.0).abs() < 1e-12);
}

/// Cosine distance between an identical row pair is zero.
#[test]
fn cosine_identical_vectors_have_zero_distance() {
    let row: Array2<f64> = array![[3.0, 4.0, 0.0]];
    let distance = sk_cosine_distance_matrix(&row.view(), &row.view());
    assert!(distance[[0, 0]].abs() < 1e-12);
}

/// Cosine distance between orthogonal unit vectors is one.
#[test]
fn cosine_orthogonal_unit_vectors_have_unit_distance() {
    let left: Array2<f64> = array![[1.0, 0.0, 0.0]];
    let right: Array2<f64> = array![[0.0, 1.0, 0.0]];
    let distance = sk_cosine_distance_matrix(&left.view(), &right.view());
    assert!((distance[[0, 0]] - 1.0).abs() < 1e-12);
}

/// Cosine distance with a scaled identical direction is zero (scale-invariant).
#[test]
fn cosine_is_scale_invariant() {
    let left: Array2<f64> = array![[1.0, 2.0]];
    let right: Array2<f64> = array![[10.0, 20.0]];
    let distance = sk_cosine_distance_matrix(&left.view(), &right.view());
    assert!(distance[[0, 0]].abs() < 1e-12);
}

/// A zero-norm row yields a cosine distance of one (denominator guard).
#[test]
fn cosine_zero_norm_row_is_defined() {
    let left: Array2<f64> = array![[0.0, 0.0]];
    let right: Array2<f64> = array![[1.0, 0.0]];
    let distance = sk_cosine_distance_matrix(&left.view(), &right.view());
    assert!((distance[[0, 0]] - 1.0).abs() < 1e-12);
}

/// Large data: a 200×20 set keeps the distance matrix symmetric, non-negative
/// and with a zero diagonal (PRD §8.7: runs with lots of data).
#[test]
fn large_data_distance_matrix_is_symmetric() {
    let points: Array2<f64> = Array2::from_shape_fn((200, 20), |(i, j)| {
        ((i * 31 + j * 17) % 101) as f64 * 0.5 - 25.0
    });
    let distance = sk_squared_euclidean_distance_matrix(&points.view(), &points.view());
    assert_eq!(distance.shape(), &[200, 200]);
    for i in 0..200 {
        assert!(distance[[i, i]].abs() < 1e-9, "nonzero diagonal at {i}");
        for j in (i + 1)..200 {
            assert!((distance[[i, j]] - distance[[j, i]]).abs() < 1e-9);
            assert!(distance[[i, j]] >= 0.0);
        }
    }
}

/// Concurrency: simultaneous computations across threads are all identical
/// (kernels are pure and thread-safe).
#[test]
fn distances_are_deterministic_under_concurrency() {
    let left: Array2<f64> = Array2::from_shape_fn((40, 5), |(i, j)| ((i * 3 + j) % 7) as f64);
    let right: Array2<f64> = Array2::from_shape_fn((10, 5), |(i, j)| ((i + j * 2) % 5) as f64);
    let expected = sk_squared_euclidean_distance_matrix(&left.view(), &right.view());

    let handles: Vec<_> = (0..8)
        .map(|_| {
            let left = left.clone();
            let right = right.clone();
            std::thread::spawn(move || {
                sk_squared_euclidean_distance_matrix(&left.view(), &right.view())
            })
        })
        .collect();
    for handle in handles {
        let result = handle.join().unwrap();
        assert_eq!(result.shape(), expected.shape());
        for ((index, value), expected_value) in result.indexed_iter().zip(expected.iter()) {
            assert!(
                (value - expected_value).abs() < 1e-12,
                "mismatch at {index:?}: {value} vs {expected_value}"
            );
        }
    }
}
