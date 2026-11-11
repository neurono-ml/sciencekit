//! DBSCAN cluster expansion (spec `dbscan`, decisions 2 and 3).

use std::collections::VecDeque;

use ndarray::Array1;
use sciencekit_common::{SKClusterIdentifier, SKClusterLabel, SKError};

/// The learned cluster partition produced by DBSCAN expansion.
#[derive(Debug, Clone)]
pub(crate) struct SKDBSCANExpansion {
    /// Typed assignment per training row (`Cluster` or `Noise`).
    pub(crate) labels: Array1<SKClusterLabel>,
    /// Training row indices of core samples, in ascending order.
    pub(crate) core_sample_indices: Array1<usize>,
    /// Number of distinct clusters discovered (excluding noise).
    pub(crate) number_of_clusters: usize,
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
    })
}
