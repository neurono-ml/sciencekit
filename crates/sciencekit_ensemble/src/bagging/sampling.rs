//! Bootstrap row and feature-subset sampling and non-densifying submatrix
//! projection for bagging estimators.

use ndarray::{Array2, ArrayView2};
use sciencekit_common::{SK_RANDOM_DOMAIN_STAGE, SKFloat, SKRandomGenerator, sk_derive_seed};
use sciencekit_tree::SKMaximumFeatures;
use sprs::{CsMat, CsMatView};

use crate::bootstrap::{SKMaximumSamples, sk_gather_sparse_rows, sk_sample_indices, sk_tree_seed};

/// Derive an independent seed for feature-subset draws of `estimator_index`.
pub(super) fn sk_feature_subset_seed(global_seed: u64, estimator_index: usize) -> u64 {
    sk_derive_seed(
        sk_tree_seed(global_seed, estimator_index),
        SK_RANDOM_DOMAIN_STAGE,
        1,
    )
}

/// Draw row indices for `estimator_index` reusing the shared ensemble sampler.
pub(super) fn sk_bagging_row_indices(
    row_count: usize,
    maximum_samples: &SKMaximumSamples<f64>,
    bootstrap: bool,
    global_seed: u64,
    estimator_index: usize,
) -> Vec<usize> {
    sk_sample_indices(
        row_count,
        maximum_samples,
        bootstrap,
        global_seed,
        estimator_index,
    )
}

/// Draw feature column indices for `estimator_index`.
pub(super) fn sk_bagging_feature_indices(
    feature_count: usize,
    maximum_features: &SKMaximumFeatures<f64>,
    bootstrap_features: bool,
    global_seed: u64,
    estimator_index: usize,
) -> Vec<usize> {
    let subset_size = maximum_features.resolved_count(feature_count);
    if feature_count == 0 || subset_size == 0 {
        return Vec::new();
    }
    let mut generator =
        SKRandomGenerator::new(sk_feature_subset_seed(global_seed, estimator_index));
    if bootstrap_features {
        return (0..subset_size)
            .map(|_| generator.next_index(feature_count))
            .collect();
    }
    if subset_size >= feature_count {
        return (0..feature_count).collect();
    }
    let mut subset = generator.sample_indices_without_replacement(feature_count, subset_size);
    subset.sort_unstable();
    subset
}

/// Whether `features` is the exact identity sequence `0..total_features`.
pub(super) fn sk_is_identity_features(features: &[usize], total_features: usize) -> bool {
    features.len() == total_features
        && features
            .iter()
            .enumerate()
            .all(|(expected_column, &column)| expected_column == column)
}

/// Gather sampled `rows` and `features` from a dense matrix.
pub(super) fn sk_gather_dense_submatrix<F: SKFloat>(
    matrix: &ArrayView2<'_, F>,
    rows: &[usize],
    features: &[usize],
) -> Array2<F> {
    Array2::from_shape_fn(
        (rows.len(), features.len()),
        |(sample_row, sample_column)| matrix[[rows[sample_row], features[sample_column]]],
    )
}

/// Project contiguous dense rows `row_start..row_end` onto `features`.
pub(super) fn sk_project_dense_rows<F: SKFloat>(
    matrix: &ArrayView2<'_, F>,
    row_start: usize,
    row_end: usize,
    features: &[usize],
) -> Array2<F> {
    Array2::from_shape_fn(
        (row_end - row_start, features.len()),
        |(row_offset, sample_column)| matrix[[row_start + row_offset, features[sample_column]]],
    )
}

/// Gather sampled `rows` and `features` from a CSR matrix without densifying.
pub(super) fn sk_gather_sparse_submatrix<F: SKFloat>(
    matrix: &CsMatView<'_, F>,
    rows: &[usize],
    features: &[usize],
) -> CsMat<F> {
    if sk_is_identity_features(features, matrix.cols()) {
        return sk_gather_sparse_rows(matrix, rows);
    }
    sk_collect_sparse_rows(matrix, rows.iter().copied(), rows.len(), features)
}

/// Project contiguous CSR rows `row_start..row_end` onto `features` without
/// densifying.
pub(super) fn sk_project_sparse_rows<F: SKFloat>(
    matrix: &CsMatView<'_, F>,
    row_start: usize,
    row_end: usize,
    features: &[usize],
) -> CsMat<F> {
    sk_collect_sparse_rows(matrix, row_start..row_end, row_end - row_start, features)
}

fn sk_collect_sparse_rows<F: SKFloat>(
    matrix: &CsMatView<'_, F>,
    row_iterator: impl Iterator<Item = usize>,
    row_count: usize,
    features: &[usize],
) -> CsMat<F> {
    let mut sorted_lookup: Vec<(usize, usize)> = features
        .iter()
        .copied()
        .enumerate()
        .map(|(local_column, source_column)| (source_column, local_column))
        .collect();
    sorted_lookup.sort_unstable();
    let ordered_without_duplicates = features.windows(2).all(|pair| pair[0] < pair[1]);

    let mut indptr = Vec::with_capacity(row_count + 1);
    let mut indices = Vec::new();
    let mut data = Vec::new();
    let mut row_entries: Vec<(usize, F)> = Vec::new();
    indptr.push(0);

    row_iterator.for_each(|source_row| {
        let view = matrix.outer_view(source_row).expect("row within bounds");
        row_entries.clear();
        view.indices()
            .iter()
            .zip(view.data().iter())
            .for_each(|(&source_column, &value)| {
                let first =
                    sorted_lookup.partition_point(|&(candidate, _)| candidate < source_column);
                sorted_lookup[first..]
                    .iter()
                    .take_while(|&&(candidate, _)| candidate == source_column)
                    .for_each(|&(_, local_column)| row_entries.push((local_column, value)));
            });
        if !ordered_without_duplicates {
            row_entries.sort_unstable_by_key(|&(local_column, _)| local_column);
        }
        row_entries.iter().for_each(|&(local_column, value)| {
            indices.push(local_column);
            data.push(value);
        });
        indptr.push(indices.len());
    });

    CsMat::new((row_count, features.len()), indptr, indices, data)
}
