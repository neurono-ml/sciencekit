//! The Calinski-Harabasz index (spec `clustering-metrics`).

use ndarray::{Array1, ArrayView1};
use sciencekit_common::{SKClusterLabel, SKDataView, SKError, SKFloat};

use super::geometry::{sk_centroid_squared_distance, sk_cluster_geometry};
use super::grouping::sk_group_clusters;
use super::inputs::sk_validate_clustering_inputs;

/// The Calinski-Harabasz index: **higher is better**.
///
/// The between-cluster dispersion (weighted centroid distance to the global
/// mean) divided by the within-cluster dispersion, each scaled by its degrees
/// of freedom. A perfectly compact clustering has zero within dispersion and
/// yields [`SKFloat::infinity`].
///
/// **Noise policy:** noise rows are excluded from every sum; fewer than two
/// clusters, or no more rows than clusters, are rejected with
/// [`SKError::InvalidHyperparameter`].
pub fn sk_calinski_harabasz_score<F: SKFloat>(
    features: SKDataView<'_, F>,
    assignments: ArrayView1<'_, SKClusterLabel>,
) -> Result<F, SKError> {
    let (_, _) = sk_validate_clustering_inputs(&features, assignments, "calinski_harabasz")?;
    let grouping = sk_group_clusters(assignments);
    let number_of_clusters = grouping.number_of_clusters();
    if number_of_clusters == 0 {
        return Err(SKError::EmptyInput {
            operation: "calinski_harabasz",
        });
    }
    if number_of_clusters < 2 {
        return Err(SKError::InvalidHyperparameter {
            name: "assignments",
            reason: "Calinski-Harabasz requires at least two clusters after excluding noise"
                .to_string(),
        });
    }
    let number_of_non_noise_rows: usize = grouping.cluster_sizes.iter().sum();
    if number_of_non_noise_rows <= number_of_clusters {
        return Err(SKError::InvalidHyperparameter {
            name: "assignments",
            reason: "Calinski-Harabasz requires more rows than clusters after excluding noise"
                .to_string(),
        });
    }

    let geometry = sk_cluster_geometry(features, assignments);
    let number_of_features = geometry.centroids.ncols();
    let mut global_mean = Array1::<F>::zeros(number_of_features);
    for (cluster, centroid) in geometry.centroids.rows().into_iter().enumerate() {
        let weight = F::from(geometry.sizes[cluster]).expect("a cluster size fits");
        for (accumulated, value) in global_mean.iter_mut().zip(centroid.iter()) {
            *accumulated = *accumulated + *value * weight;
        }
    }
    let total_weight = F::from(number_of_non_noise_rows).expect("a row count fits");
    for value in global_mean.iter_mut() {
        *value = *value / total_weight;
    }

    let mut between = F::zero();
    for (cluster, centroid) in geometry.centroids.rows().into_iter().enumerate() {
        let weight = F::from(geometry.sizes[cluster]).expect("a cluster size fits");
        between = between + weight * sk_centroid_squared_distance(centroid, global_mean.view());
    }
    let within = geometry
        .squared_scatter
        .iter()
        .fold(F::zero(), |total, scatter| total + *scatter);
    if within.is_zero() {
        return Ok(F::infinity());
    }

    let between_degrees = F::from(number_of_clusters - 1).expect("a cluster count fits");
    let within_degrees =
        F::from(number_of_non_noise_rows - number_of_clusters).expect("a row count fits");
    Ok((between / between_degrees) / (within / within_degrees))
}
