//! Subsample views and seeded per-tree draw helpers for isolation tree growth.

use ndarray::ArrayView2;
use sciencekit_common::{
    SK_RANDOM_DOMAIN_STAGE, SK_RANDOM_DOMAIN_TREE, SKFloat, SKRandomGenerator, sk_derive_seed,
};

/// Representation-neutral subsample interface for growing an isolation tree.
pub(crate) trait IsolationSampleSource<F: SKFloat> {
    /// Number of sampled rows in this subsample.
    fn sample_count(&self) -> usize;

    /// Number of columns in the feature space.
    fn feature_count(&self) -> usize;

    /// Observed `(minimum, maximum)` of `feature_index` across `rows` (which is non-empty).
    fn feature_bounds(&self, rows: &[usize], feature_index: usize) -> (F, F);

    /// Splits `rows` into `(left, right)` using `value <= threshold` for left.
    fn partition_rows(
        &self,
        rows: &[usize],
        feature_index: usize,
        threshold: F,
    ) -> (Vec<usize>, Vec<usize>);
}

/// Borrowed dense subsample matrix.
pub(crate) struct DenseSubsample<'a, F> {
    matrix: ArrayView2<'a, F>,
}

impl<'a, F: SKFloat> DenseSubsample<'a, F> {
    /// Wraps a dense subsample view.
    pub(crate) fn new(matrix: ArrayView2<'a, F>) -> Self {
        Self { matrix }
    }
}

impl<F: SKFloat> IsolationSampleSource<F> for DenseSubsample<'_, F> {
    fn sample_count(&self) -> usize {
        self.matrix.nrows()
    }

    fn feature_count(&self) -> usize {
        self.matrix.ncols()
    }

    fn feature_bounds(&self, rows: &[usize], feature_index: usize) -> (F, F) {
        let first_value = self.matrix[[rows[0], feature_index]];
        rows.iter().skip(1).fold(
            (first_value, first_value),
            |(current_minimum, current_maximum), &row_index| {
                let value = self.matrix[[row_index, feature_index]];
                let next_minimum = if value < current_minimum {
                    value
                } else {
                    current_minimum
                };
                let next_maximum = if value > current_maximum {
                    value
                } else {
                    current_maximum
                };
                (next_minimum, next_maximum)
            },
        )
    }

    fn partition_rows(
        &self,
        rows: &[usize],
        feature_index: usize,
        threshold: F,
    ) -> (Vec<usize>, Vec<usize>) {
        rows.iter()
            .copied()
            .partition(|&row_index| self.matrix[[row_index, feature_index]] <= threshold)
    }
}

/// Resolves `maximum_features` in `(0, 1]` to a feature draw count in `1..=columns_count`.
pub(super) fn sk_resolve_feature_count<F: SKFloat>(
    maximum_features: F,
    columns_count: usize,
) -> usize {
    let scaled = maximum_features * F::from(columns_count).expect("column count fits scalar");
    scaled
        .floor()
        .to_usize()
        .unwrap_or(1)
        .clamp(1, columns_count)
}

/// Produces `(sampled_rows, candidate_features, split_generator)` from independent sub-seeds.
pub(super) fn sk_prepare_tree_draws(
    global_seed: u64,
    tree_index: usize,
    rows_count: usize,
    subsample_size: usize,
    bootstrap: bool,
    columns_count: usize,
    feature_draw_count: usize,
) -> (Vec<usize>, Vec<usize>, SKRandomGenerator) {
    let tree_seed = sk_derive_seed(global_seed, SK_RANDOM_DOMAIN_TREE, tree_index as u64);
    let subsample_seed = sk_derive_seed(tree_seed, SK_RANDOM_DOMAIN_STAGE, 0);
    let feature_seed = sk_derive_seed(tree_seed, SK_RANDOM_DOMAIN_STAGE, 1);
    let split_seed = sk_derive_seed(tree_seed, SK_RANDOM_DOMAIN_STAGE, 2);

    let sampled_rows = if !bootstrap {
        if subsample_size >= rows_count {
            (0..rows_count).collect()
        } else {
            SKRandomGenerator::new(subsample_seed)
                .sample_indices_without_replacement(rows_count, subsample_size)
        }
    } else {
        let mut row_generator = SKRandomGenerator::new(subsample_seed);
        (0..subsample_size)
            .map(|_| row_generator.next_index(rows_count))
            .collect()
    };

    let candidate_features = if feature_draw_count >= columns_count {
        (0..columns_count).collect()
    } else {
        let mut chosen = SKRandomGenerator::new(feature_seed)
            .sample_indices_without_replacement(columns_count, feature_draw_count);
        chosen.sort_unstable();
        chosen
    };

    (
        sampled_rows,
        candidate_features,
        SKRandomGenerator::new(split_seed),
    )
}
