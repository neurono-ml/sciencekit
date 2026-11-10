//! The Davies-Bouldin index (spec `clustering-metrics`).

use ndarray::ArrayView1;
use sciencekit_common::{SKClusterLabel, SKDataView, SKError, SKFloat};

use super::geometry::{sk_centroid_squared_distance, sk_cluster_geometry};
use super::grouping::sk_group_clusters;
use super::inputs::sk_validate_clustering_inputs;

/// The Davies-Bouldin index: **lower is better**.
///
/// For every cluster, the worst ratio `(scatter_i + scatter_j) / distance(i, j)`
/// over the other clusters is averaged. Scatter is the mean Euclidean distance
/// of a cluster's rows to its centroid. Coincident centroids yield an infinite
/// ratio, reported as [`SKFloat::infinity`].
///
/// **Noise policy:** noise rows are excluded; fewer than two non-noise clusters
/// are rejected with [`SKError::InvalidHyperparameter`].
pub fn sk_davies_bouldin_score<F: SKFloat>(
    features: SKDataView<'_, F>,
    assignments: ArrayView1<'_, SKClusterLabel>,
) -> Result<F, SKError> {
    sk_validate_clustering_inputs(&features, assignments, "davies_bouldin")?;
    let number_of_clusters = sk_group_clusters(assignments).number_of_clusters();
    if number_of_clusters == 0 {
        return Err(SKError::EmptyInput {
            operation: "davies_bouldin",
        });
    }
    if number_of_clusters < 2 {
        return Err(SKError::InvalidHyperparameter {
            name: "assignments",
            reason: "Davies-Bouldin requires at least two clusters after excluding noise"
                .to_string(),
        });
    }

    let geometry = sk_cluster_geometry(features, assignments);
    let mut total = F::zero();
    for own in 0..number_of_clusters {
        let mut worst_ratio = F::zero();
        let own_centroid = geometry.centroids.row(own);
        for other in 0..number_of_clusters {
            if own == other {
                continue;
            }
            let distance =
                sk_centroid_squared_distance(own_centroid, geometry.centroids.row(other)).sqrt();
            let ratio = if distance.is_zero() {
                F::infinity()
            } else {
                (geometry.mean_scatter[own] + geometry.mean_scatter[other]) / distance
            };
            if ratio > worst_ratio {
                worst_ratio = ratio;
            }
        }
        total = total + worst_ratio;
    }
    Ok(total / F::from(number_of_clusters).expect("a cluster count fits in the scalar"))
}
