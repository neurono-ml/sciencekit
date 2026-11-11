//! Seeded per-tree bootstrap sampling and row gathering without densification.

use ndarray::{Array2, ArrayView2};
use sciencekit_common::{
    SK_RANDOM_DOMAIN_STAGE, SK_RANDOM_DOMAIN_TREE, SKFloat, SKRandomGenerator, sk_derive_seed,
};
use sprs::{CsMat, CsMatView};

use super::maximum_samples::SKMaximumSamples;

/// Derive the deterministic split seed of `tree_index` from `global_seed`.
pub(crate) fn sk_tree_seed(global_seed: u64, tree_index: usize) -> u64 {
    sk_derive_seed(global_seed, SK_RANDOM_DOMAIN_TREE, tree_index as u64)
}

/// Derive the deterministic bootstrap seed of `tree_index` from `global_seed`.
///
/// Uses a distinct domain over the tree seed so row sampling never shares a
/// generator stream with the tree's per-node feature draws.
pub(crate) fn sk_bootstrap_seed(global_seed: u64, tree_index: usize) -> u64 {
    sk_derive_seed(
        sk_tree_seed(global_seed, tree_index),
        SK_RANDOM_DOMAIN_STAGE,
        0,
    )
}

/// Draw row indices for `tree_index` according to `maximum_samples` and
/// `bootstrap`.
pub(crate) fn sk_sample_indices<F: SKFloat>(
    row_count: usize,
    maximum_samples: &SKMaximumSamples<F>,
    bootstrap: bool,
    global_seed: u64,
    tree_index: usize,
) -> Vec<usize> {
    let sample_size = maximum_samples.resolved_count(row_count);
    if row_count == 0 || sample_size == 0 {
        return Vec::new();
    }
    if !bootstrap {
        if sample_size >= row_count {
            return (0..row_count).collect();
        }
        let mut generator = SKRandomGenerator::new(sk_bootstrap_seed(global_seed, tree_index));
        return generator.sample_indices_without_replacement(row_count, sample_size);
    }
    let mut generator = SKRandomGenerator::new(sk_bootstrap_seed(global_seed, tree_index));
    (0..sample_size)
        .map(|_| generator.next_index(row_count))
        .collect()
}

/// Materialize a contiguous dense matrix containing the sampled rows.
pub(crate) fn sk_gather_dense_rows<F: SKFloat>(
    matrix: &ArrayView2<'_, F>,
    rows: &[usize],
) -> Array2<F> {
    Array2::from_shape_fn((rows.len(), matrix.ncols()), |(sample_row, column)| {
        matrix[[rows[sample_row], column]]
    })
}

/// Materialize a CSR matrix containing the sampled rows without densifying.
pub(crate) fn sk_gather_sparse_rows<F: SKFloat>(
    matrix: &CsMatView<'_, F>,
    rows: &[usize],
) -> CsMat<F> {
    let non_zero_count: usize = rows
        .iter()
        .map(|&row| matrix.outer_view(row).expect("row within bounds").nnz())
        .sum();
    let mut indptr = Vec::with_capacity(rows.len() + 1);
    let mut indices = Vec::with_capacity(non_zero_count);
    let mut data = Vec::with_capacity(non_zero_count);
    indptr.push(0);
    rows.iter().for_each(|&row| {
        let view = matrix.outer_view(row).expect("row within bounds");
        indices.extend_from_slice(view.indices());
        data.extend_from_slice(view.data());
        indptr.push(indices.len());
    });
    CsMat::new((rows.len(), matrix.cols()), indptr, indices, data)
}
