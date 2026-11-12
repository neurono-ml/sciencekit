//! Seeded feature subsampling for one node.

use sciencekit_common::{SK_RANDOM_DOMAIN_TREE, SKRandomGenerator, sk_derive_seed};

use super::SKGrowthConfig;

/// The features one node considers, in ascending order.
///
/// With `All` the full range is returned without consuming the seeded stream;
/// otherwise the resolved count is drawn without replacement and sorted so the
/// scan order stays deterministic.
pub(crate) fn sk_select_features<F: sciencekit_common::SKFloat>(
    config: &SKGrowthConfig<F>,
    feature_count: usize,
    node_index: usize,
) -> Vec<usize> {
    let resolved = config.maximum_features.resolved_count(feature_count);
    if resolved >= feature_count {
        return (0..feature_count).collect();
    }
    let seed = sk_derive_seed(config.random_seed, SK_RANDOM_DOMAIN_TREE, node_index as u64);
    let mut generator = SKRandomGenerator::new(seed);
    let mut features = generator.sample_indices_without_replacement(feature_count, resolved);
    features.sort_unstable();
    features
}
