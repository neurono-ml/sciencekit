//! DBSCAN cluster expansion (spec `dbscan`, decisions 2 and 3).

use std::collections::VecDeque;

use ndarray::{Array1, Axis};
use sciencekit_common::{SKClusterIdentifier, SKClusterLabel, SKError, SKFloat};
use sciencekit_neighbors::SKTrainingData;
use sprs::CsMat;

use super::config::{SKDBSCANConfiguration, SKDBSCANSearchStrategy};
use super::region_query::{sk_build_index, sk_radius_neighborhoods, sk_resolve_strategy};

/// The learned cluster partition produced by DBSCAN expansion.
#[derive(Debug, Clone)]
pub(crate) struct SKDBSCANExpansion {
    /// Typed assignment per training row (`Cluster` or `Noise`).
    pub(crate) labels: Array1<SKClusterLabel>,
    /// Training row indices of core samples, in ascending order.
    pub(crate) core_sample_indices: Array1<usize>,
    /// Number of distinct clusters discovered (excluding noise).
    pub(crate) number_of_clusters: usize,
    /// Search strategy resolved for the region query (never `Auto`).
    pub(crate) strategy: SKDBSCANSearchStrategy,
}

/// Resolve the search strategy, run the inclusive `eps` region query and expand
/// the density-connected components deterministically.
pub(crate) fn sk_run_dbscan_partition<F: SKFloat>(
    configuration: &SKDBSCANConfiguration<F>,
    training: &SKTrainingData<F>,
) -> Result<SKDBSCANExpansion, SKError> {
    let strategy = sk_resolve_strategy(
        configuration.algorithm,
        configuration.metric,
        training.is_dense(),
        training.columns(),
    )?;
    let index = sk_build_index(configuration, training, strategy)?;
    let neighborhoods = sk_radius_neighborhoods(&index, training, configuration.eps)?;
    sk_expand_clusters(&neighborhoods, configuration.minimum_samples, strategy)
}

/// Expand density-connected components from pre-computed `eps` neighbourhoods.
///
/// Core samples are training rows whose inclusive `eps` neighbourhood contains
/// at least `minimum_samples` points (the point itself included). Components
/// are seeded in ascending training-index order so cluster identifiers are
/// deterministic and border points receive the cluster that reaches them first.
pub(crate) fn sk_expand_clusters(
    neighborhoods: &[Vec<usize>],
    minimum_samples: usize,
    strategy: SKDBSCANSearchStrategy,
) -> Result<SKDBSCANExpansion, SKError> {
    let is_core: Vec<bool> = neighborhoods
        .iter()
        .map(|neighbors| neighbors.len() >= minimum_samples)
        .collect();
    let core_sample_indices = Array1::from_iter(
        is_core
            .iter()
            .enumerate()
            .filter_map(|(row, &core)| core.then_some(row)),
    );

    let mut labels = vec![SKClusterLabel::Noise; neighborhoods.len()];
    let mut queue = VecDeque::new();
    let mut next_cluster_id: usize = 0;

    for (seed, &seed_is_core) in is_core.iter().enumerate() {
        if !seed_is_core || !labels[seed].is_noise() {
            continue;
        }
        if next_cluster_id > SKClusterIdentifier::MAXIMUM_INDEX as usize {
            return Err(SKError::Conversion(format!(
                "cluster count exceeds maximum supported identifier {}",
                SKClusterIdentifier::MAXIMUM_INDEX
            )));
        }
        let cluster_label = SKClusterLabel::cluster(next_cluster_id as u32);
        next_cluster_id += 1;
        labels[seed] = cluster_label;
        queue.push_back(seed);

        while let Some(current) = queue.pop_front() {
            for &neighbor in &neighborhoods[current] {
                if labels[neighbor].is_noise() {
                    labels[neighbor] = cluster_label;
                    if is_core[neighbor] {
                        queue.push_back(neighbor);
                    }
                }
            }
        }
    }

    Ok(SKDBSCANExpansion {
        labels: Array1::from_vec(labels),
        core_sample_indices,
        number_of_clusters: next_cluster_id,
        strategy,
    })
}

/// Extract the core rows from `training` while preserving its representation
/// (dense `Array2` or sparse CSR `CsMat`) without densifying sparse storage.
pub(crate) fn sk_extract_core_samples<F: SKFloat>(
    training: &SKTrainingData<F>,
    core_indices: &[usize],
) -> SKTrainingData<F> {
    match training {
        SKTrainingData::Dense(dense) => SKTrainingData::Dense(dense.select(Axis(0), core_indices)),
        SKTrainingData::Sparse(sparse) => {
            let mut indptr = Vec::with_capacity(core_indices.len() + 1);
            let mut indices = Vec::new();
            let mut values = Vec::new();
            indptr.push(0);
            for &row in core_indices {
                if let Some(vector) = sparse.outer_view(row) {
                    indices.extend_from_slice(vector.indices());
                    values.extend_from_slice(vector.data());
                }
                indptr.push(indices.len());
            }
            SKTrainingData::Sparse(CsMat::new(
                (core_indices.len(), sparse.cols()),
                indptr,
                indices,
                values,
            ))
        }
    }
}
