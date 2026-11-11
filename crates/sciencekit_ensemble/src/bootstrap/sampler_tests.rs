//! Companion tests for seeded per-tree bootstrap sampling (task 1.2).

use std::collections::HashSet;

use ndarray::array;
use sprs::CsMat;

use super::{
    SKMaximumSamples, sk_bootstrap_seed, sk_gather_dense_rows, sk_gather_sparse_rows,
    sk_sample_indices, sk_tree_seed,
};

/// The same global seed and tree index always produce the exact same sample,
/// while distinct tree indices diverge and never collide with the split seed.
#[test]
fn bootstrap_samples_are_reproducible_per_tree() {
    let first = sk_sample_indices(32, &SKMaximumSamples::<f64>::All, true, 42, 3);
    let second = sk_sample_indices(32, &SKMaximumSamples::<f64>::All, true, 42, 3);
    let other_tree = sk_sample_indices(32, &SKMaximumSamples::<f64>::All, true, 42, 4);
    assert_eq!(first, second);
    assert_ne!(first, other_tree);
    assert_ne!(sk_tree_seed(42, 3), sk_bootstrap_seed(42, 3));
}

/// `All`, `Count` and `Fraction` resolve the expected sample sizes.
#[test]
fn bootstrap_sample_sizes_follow_maximum_samples_policy() {
    let all = sk_sample_indices(20, &SKMaximumSamples::<f64>::All, true, 7, 0);
    let count = sk_sample_indices(20, &SKMaximumSamples::<f64>::Count(6), true, 7, 0);
    let clamped = sk_sample_indices(20, &SKMaximumSamples::<f64>::Count(50), true, 7, 0);
    let fraction = sk_sample_indices(20, &SKMaximumSamples::Fraction(0.25_f64), true, 7, 0);
    assert_eq!(all.len(), 20);
    assert_eq!(count.len(), 6);
    assert_eq!(clamped.len(), 20);
    assert_eq!(fraction.len(), 5);
    assert!(all.iter().all(|&row| row < 20));
}

/// Sampling with replacement repeats at least one row on a full-size draw.
#[test]
fn bootstrap_draws_with_replacement() {
    let drawn = sk_sample_indices(64, &SKMaximumSamples::<f64>::All, true, 19, 0);
    let unique: HashSet<usize> = drawn.iter().copied().collect();
    assert_eq!(drawn.len(), 64);
    assert!(
        unique.len() < drawn.len(),
        "with-replacement bootstrap of 64 rows should repeat at least one index"
    );
}

/// Disabling bootstrap over `All` returns every row in order without replacement.
#[test]
fn no_bootstrap_returns_all_rows_without_duplicates() {
    let drawn = sk_sample_indices(10, &SKMaximumSamples::<f64>::All, false, 99, 2);
    assert_eq!(drawn, (0..10).collect::<Vec<usize>>());

    let subsampled = sk_sample_indices(10, &SKMaximumSamples::<f64>::Count(4), false, 99, 2);
    let unique: HashSet<usize> = subsampled.iter().copied().collect();
    assert_eq!(subsampled.len(), 4);
    assert_eq!(unique.len(), 4);
}

/// Dense and sparse row gathers preserve duplicates, order and sparsity.
#[test]
fn row_gathers_preserve_order_and_sparsity() {
    let dense = array![[1.0, 0.0], [0.0, 2.0], [3.0, 4.0]];
    let sparse = CsMat::new(
        (3, 2),
        vec![0, 1, 2, 4],
        vec![0, 1, 0, 1],
        vec![1.0, 2.0, 3.0, 4.0],
    );
    let rows = [2, 0, 2];
    let gathered_dense = sk_gather_dense_rows(&dense.view(), &rows);
    let gathered_sparse = sk_gather_sparse_rows(&sparse.view(), &rows);
    assert_eq!(gathered_dense, array![[3.0, 4.0], [1.0, 0.0], [3.0, 4.0]]);
    assert_eq!(gathered_sparse.shape(), (3, 2));
    assert_eq!(gathered_sparse.nnz(), 5);
    assert_eq!(gathered_sparse.to_dense(), gathered_dense);
}
